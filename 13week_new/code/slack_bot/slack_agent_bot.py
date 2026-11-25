"""[13주차 · 미니 프로젝트] ★★ 슬랙 봇 — 3교시 19장 «오늘 것을 붙이는 세 자리» 가 실제로 도는 봇

   10주차 slack_bot_mode/slack_rag_bot.py 에서 출발했습니다. 입구 · 출구(슬랙)는 그대로, 두뇌만 바뀌었습니다.

       10주차   rag_backend.ask()                    → 언제나 검색 · 한 번 묻고 끝
       13주차   bot_graph.Brain  (LangGraph 그래프)   → 분류 → ② Agentic RAG / ③ MCP 도구 + 승인
                                                       ① 같은 스레드는 이어서 기억

   전체 흐름

     슬랙 @봇 질문 ══ WebSocket (xapp) ══▶ 내 PC  slack_agent_bot.py
                                                  │  ① thread_id = f"{channel}-{thread_ts}"
                                                  ▼
                                          bot_graph.Brain.ask()      ← 그래프 + 체크포인터 + MCP 서버
                                                  │
                         ┌────────────────────────┴───────────────────────┐
                    끝까지 감                                          ⏸ interrupt (쓰기·삭제 도구)
                         │                                                │
     슬랙 스레드 ◀── 답 + 📚 근거 + 경로                    «승인이 필요합니다» 를 스레드에 올리고 함수 끝
                                                                          │   (봇은 아무도 기다리지 않는다)
                                                     몇 분 뒤 같은 스레드에 «@봇 승인» / «@봇 거부»
                                                                          ▼
                                                          Brain.resume() = Command(resume=…) → 이어서 실행

   슬랙에서 쓰는 법
     @봇 몇 주를 결석하면 제적되나요?        → ② 학칙에서 찾아 답 + 📚 근거
     (같은 스레드에서) @봇 그럼 휴학은요?     → ① 앞 질문을 기억해서 «휴학» 질문으로 풀어 답
     @봇 메모 남겨줘: 금요일 휴학 상담        → ③ ⏸ 승인 요청 → 같은 스레드에 @봇 승인 / @봇 거부
     @봇 내 메모 보여줘                       → ③ 읽기 도구라 승인 없이 바로
     @봇 검색 질문                            → (10주차 그대로) 어떤 조항이 뽑혔는지만
     @봇 도움말                               → 사용법

  ⚠️ 같은 슬랙 앱(xapp)으로 10주차 봇을 동시에 켜 두면 이벤트가 둘 중 한쪽으로 갈린다 — 하나만 켠다.
  ⚠️ 승인은 «요청한 사람» 또는 .env 의 SLACK_ALLOWED_USERS 만 할 수 있다 (남이 대신 누르는 것 방지).
  ⚠️ 체크포인터가 InMemorySaver 라서 봇을 껐다 켜면 스레드 기억 · 승인 대기가 모두 사라진다 (2교시 8장).

전제 (README.md «실행 순서»):
    python bot_graph.py 로 터미널에서 먼저 답이 나와야 한다 — 봇이 이상하면 범인을 가르는 기준점 ★

실행 (이 폴더에서):
    python slack_agent_bot.py            # 끝낼 때는 Ctrl+C
    python slack_agent_bot.py --no-mcp   # MCP 없이 (RAG 만 하는 팀)
"""

from dotenv import load_dotenv

load_dotenv()  # ★ 맨 위 — LANGSMITH_* 를 먼저 읽힌다

import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

from slack_bolt import App  # noqa: E402
from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: E402
from slack_sdk.errors import SlackApiError  # noqa: E402

# ★ 10주차와 다른 곳 — 두뇌가 그래프(Brain)로 바뀌었다
import bot_config as C  # noqa: E402  ✏️ 내 데이터 봇 설정 — 이 파일은 고치지 않아도 된다
from bot_graph import DANGEROUS, EMBED_MODEL, INDEX_DIR, MODEL, ROUTES, TOOL_MODEL, Brain, source_of  # noqa: E402
from bot_graph import search_raw as search  # noqa: E402  ← «검색» 모드 — 10주차와 같은 일, 인덱스는 bot_config.INDEX

SEARCH_WORDS = ("검색", "search")
HELP_WORDS = ("도움말", "help", "사용법")
APPROVE_WORDS = ("승인", "approve", "yes", "ㅇㅇ")
REJECT_WORDS = ("거부", "reject", "no", "취소")
MAX_QUESTION = 300  # 9주차 2교시 «입력 검증»

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
ALLOWED = {u.strip() for u in os.environ.get("SLACK_ALLOWED_USERS", "").split(",") if u.strip()}

if not BOT_TOKEN.startswith("xoxb-"):
    raise SystemExit("SLACK_BOT_TOKEN 은 xoxb- 로 시작해야 합니다.")
if not APP_TOKEN.startswith("xapp-"):
    raise SystemExit("SLACK_APP_TOKEN 은 xapp- 로 시작해야 합니다.")

app = App(token=BOT_TOKEN)
brain: Brain | None = None  # __main__ 에서 start()

USAGE = (
    f"{C.BOT_INTRO}입니다 🙂\n"
    + (f"• `@봇 질문` — {C.DOC_NAME}에서 찾아 근거와 함께 답합니다\n" if C.USE_RAG else "")
    + "• 같은 스레드에서 이어 물으면 앞 대화를 기억합니다 (`@봇 그럼 그건요?`)\n"
    + ("• `@봇 메모 남겨줘: …` · `@봇 내 메모 보여줘` · `@봇 2번 메모 지워줘` — 저장 · 삭제는 승인 후 실행\n"
       if C.USE_MCP else "")
    + ("• `@봇 검색 질문` — 어떤 문서가 뽑혔는지만" if C.USE_RAG else "")
)


# ── 1. 메시지 → 글자 ─────────────────────────────────────────────
def clean(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text or "").strip()  # 앞의 <@U07ABCDEF> 를 떼어 낸다


def to_mrkdwn(answer: str) -> str:
    """LLM 은 **굵게** 로 쓰고, 슬랙은 *굵게* 로 읽는다."""
    return answer.strip().replace("**", "*")


def search_text(question: str) -> str:
    lines = [f"🔎 *검색 결과* — 거리(작을수록 가깝다) · `{EMBED_MODEL}`"]
    for i, (doc, dist) in enumerate(search(question), 1):
        preview = doc.page_content[:80].replace("\n", " ")
        lines.append(f"{i}. `{dist:.3f}` *{source_of(doc)}*\n> {preview}…")
    return "\n".join(lines)


def approval_text(req: dict) -> str:
    calls = "\n".join(f"• `{c['name']}({c['args']})`" for c in req.get("calls", []))
    who = f"<@{req['requested_by']}>" if req.get("requested_by") else "요청한 사람"
    return (f"⏸ *승인이 필요합니다* — 되돌릴 수 없는 도구입니다\n{calls}\n"
            f"이 스레드에 `@봇 승인` 또는 `@봇 거부` 로 답해 주세요. ({who} 또는 허용된 사람만)")


def result_text(r: dict) -> str:
    if r["status"] == "waiting":
        return approval_text(r["approval"])
    tail = f"\n\n📚 *근거*  {'  /  '.join(r['sources'])}" if r.get("sources") else ""
    return (f"{to_mrkdwn(r['answer'])}{tail}\n"
            f"_— 경로: {' → '.join(r['path'])} · {r['sec']:.1f}초_")


# ── 2. 답하기 — 자리부터 잡고, 답이 오면 고친다 (9 · 10주차 그대로) ──────
def answer(event: dict, client) -> None:
    channel = event["channel"]
    thread = event.get("thread_ts") or event["ts"]
    user = event.get("user", "")
    text = clean(event.get("text", ""))
    thread_id = f"{channel}-{thread}"  # ① ★ 스레드마다 기억 — 슬랙 스레드 하나 = LangGraph thread 하나
    meta = {"slack_user": user, "slack_channel": channel}
    print(f"받음 ← user={user} thread={thread_id} {text!r}")

    def post(msg: str):
        return client.chat_postMessage(channel=channel, thread_ts=thread, text=msg)

    if not text or text.lower() in HELP_WORDS:
        post(USAGE)
        return
    if len(text) > MAX_QUESTION:
        post(f"✋ 질문은 {MAX_QUESTION}자 이내로 부탁드립니다. (지금 {len(text)}자)")
        return

    first, _, rest = text.partition(" ")
    if first.lower() in SEARCH_WORDS and C.USE_RAG:  # 10주차 «검색» 모드 — LLM 없이 즉시
        try:
            post(search_text(rest.strip() or C.TOPIC))
        except Exception as e:
            post(f"⚠️ 검색 실패 — `{type(e).__name__}: {str(e)[:200]}`")
        return

    # ── ⏸ 이 스레드에 승인 대기 중인 요청이 있으면 — 이번 메시지는 «답» 이다 ──
    req = brain.pending(thread_id)
    if req is not None:
        word = text.split()[0].lower()
        if word not in APPROVE_WORDS + REJECT_WORDS:
            post("⏸ 이 스레드에 승인 대기 중인 요청이 있습니다. `@봇 승인` 또는 `@봇 거부` 로 먼저 답해 주세요.\n"
                 + approval_text(req))
            return
        if user != req.get("requested_by") and user not in ALLOWED:  # 남이 대신 승인하지 못하게
            post(f"🔒 승인은 <@{req.get('requested_by')}> 님 또는 허용된 사람만 할 수 있습니다.")
            return
        decision = "approve" if word in APPROVE_WORDS else "reject"
        holder = post(f"⏳ {'승인' if decision == 'approve' else '거부'} — 이어서 실행합니다…")
        try:
            r = brain.resume(thread_id, decision, meta)  # ★ Command(resume=…) — 같은 thread_id 로 재개
            msg = result_text(r)
            print(f"  재개 → {decision} · {' → '.join(r['path'])}")
        except Exception as e:
            msg = f"⚠️ 재개 실패 — `{type(e).__name__}: {str(e)[:200]}`"
            print(f"  실패 → {type(e).__name__}: {e}")
        client.chat_update(channel=channel, ts=holder["ts"], text=msg)
        return

    # ── 새 질문 — ① 자리 잡기 → ② 그래프 실행 → ③ 자리를 답(또는 승인 요청)으로 ★ ──
    holder = post(f"⏳ 생각 중… ({MODEL})")
    try:
        r = brain.ask(thread_id, text, user=user, meta=meta)
        msg = result_text(r)
        print(f"  {r['status']} → {' → '.join(r['path'])} · {r['sec']:.1f}초 · 스레드 메시지 {r['n_messages']}개")
    except Exception as e:
        msg = f"⚠️ 실패 — `{type(e).__name__}: {str(e)[:200]}`"
        print(f"  실패 → {type(e).__name__}: {e}")
    client.chat_update(channel=channel, ts=holder["ts"], text=msg)


# ── 3. 이벤트 연결 (9 · 10주차 그대로) ─────────────────────────────
@app.event("app_mention")
def on_mention(event, client):
    answer(event, client)


@app.event("message")
def on_dm(event, client):
    # (선택) 봇에게 1:1 DM 으로 물을 때 — message.im 이벤트를 구독해야 온다
    if event.get("channel_type") == "im" and not event.get("subtype"):
        answer(event, client)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print(f"봇         : {C.BOT_INTRO}  (갈래 {' · '.join(ROUTES)})")
    print(f"인덱스     : {INDEX_DIR}")
    print(f"모델       : {MODEL} (판단 · 답)  /  도구 모델 {TOOL_MODEL}")
    if C.USE_RAG:
        try:
            top = search(C.TOPIC, k=1)[0][0]  # 켜기 전에 인덱스를 한 번 읽어 본다 (10주차와 같은 점검)
            print(f"인덱스 확인 ✅ ({source_of(top)})")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"인덱스 검색 실패 — {type(e).__name__}: {e}\n"
                             "  → ollama 가 켜져 있나?  EMBED_MODEL 이 인덱스를 만든 모델과 같은가?")

    brain = Brain(use_mcp=C.USE_MCP and "--no-mcp" not in sys.argv).start()
    print(f"MCP 도구   : {', '.join(brain.tool_names) or '(끔)'}  · 승인 필요: {', '.join(sorted(DANGEROUS))}")
    print(f"승인 허용  : 요청한 사람{' + ' + ', '.join(sorted(ALLOWED)) if ALLOWED else ''}")
    print("슬랙에서 @내앱이름 질문.  끝낼 때는 Ctrl+C")
    print("=" * 60)
    try:
        SocketModeHandler(app, APP_TOKEN).start()
    except SlackApiError as e:
        raise SystemExit(f"앱 토큰으로 연결하지 못했습니다: {e.response['error']}  → 9주차 실습 1 socket_hello.py 부터 다시")
    finally:
        brain.close()
