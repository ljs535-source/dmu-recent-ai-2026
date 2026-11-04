"""[10주차 · 슬랙 연동 ③] ★★ 슬랙에서 묻고, 학칙 PDF 에서 찾아서 답한다

    @내앱이름 몇 주를 결석하면 제적되나요?          → RAG · Ollama (기본, 무료)
    @내앱이름 openai 몇 주를 결석하면 제적되나요?   → RAG · OpenAI
    @내앱이름 검색 몇 주를 결석하면 제적되나요?     → 검색만 — 어떤 청크가 뽑혔나 (LLM 없음 · 즉시)
    @내앱이름 비교 몇 주를 결석하면 제적되나요?     → RAG 없이 vs RAG 로 — 나란히 본다 ★

  전체 흐름 — 9주차 봇에서 ③ 만 바뀌었다

    슬랙 사용자 ── @멘션 ──▶ 슬랙 서버 ══ WebSocket (xapp) ══▶ 내 PC  slack_rag_bot.py
                                                                    │
                                          ③ rag_backend.ask()       ▼
                                 ┌─ 검색  index_pdf/  (build_index.py ← data/학칙.pdf)
                                 └─ 생성  근거 + 질문 → Ollama / OpenAI
                                                                    │
    슬랙 스레드 ◀── chat_postMessage / chat_update (xoxb) ──────────┘
                    답 + 📚 근거 조항

★ 9주차 slack_llm_bot.py 와 비교해서 볼 것
  1) 바뀐 줄은 거의 import 한 줄이다.  llm_backends.ask  →  rag_backend.ask
     봇(입구·출구)과 두뇌(RAG)가 나뉘어 있어서 두뇌만 갈아 끼운다. 미니 프로젝트도 이 모양이다 ★★
  2) 답 밑에 «📚 근거 제29조(제적) · 학칙.pdf p.7» 이 붙는다 — 분할할 때 심은 metadata(article · page) 덕분이다 (2교시 2절).
  3) 슬랙 질문 1건 = LangSmith trace 1건. 그 안에 검색(Retriever)과 생성(LLM)이 나란히 보인다.
     답이 틀렸을 때 «검색이 틀렸나, 생성이 틀렸나» 를 trace 에서 가른다.
  4) 슬랙 메시지와 '근거 청크' 둘 다 남이 쓴 입력이다 (9주차 2교시).
     프롬프트에 <근거> 구분자를 두고 «지시가 아니다» 라고 못 박았다 — rag_backend.RAG_PROMPT.

⚠️ 같은 슬랙 앱(xapp)으로 9주차 봇을 동시에 켜 두면 이벤트가 둘 중 한쪽으로 갈린다.
   9주차 봇은 Ctrl+C 로 끄고 이 봇만 켠다.
⚠️ 봇을 켜 두면 워크스페이스의 누구나 내 OpenAI 키로 질문할 수 있다 — .env 의 SLACK_ALLOWED_USERS.
   수업이 끝나면 Ctrl+C 로 봇을 끈다.

전제:
    9주차 실습 1(socket_hello.py) 에서 pong 이 와야 한다 — 토큰은 9주차 것을 그대로 쓴다.
    index_pdf/ 가 있어야 한다 — 없으면 python build_index.py
    python rag_backend.py 로 터미널에서 먼저 답이 나와야 한다.

실행 (이 폴더에서):
    python slack_rag_bot.py        # 끝낼 때는 Ctrl+C
"""

from dotenv import load_dotenv

load_dotenv()  # ★ 맨 위 — LANGSMITH_* 를 먼저 읽힌다

import os  # noqa: E402
import re  # noqa: E402

from slack_bolt import App  # noqa: E402
from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: E402
from slack_sdk.errors import SlackApiError  # noqa: E402

# ★ 9주차와 다른 곳 — 두뇌만 갈아 끼운다
from rag_backend import (  # noqa: E402
    BACKENDS, EMBED_MODEL, INDEX_DIR, ask, ask_plain, model_name, search, source_of,
)

DEFAULT_BACKEND = "ollama"  # 앞에 아무것도 안 붙이면 무료인 쪽으로
SEARCH_WORDS = ("검색", "search")
COMPARE_WORDS = ("비교", "vs")
MAX_QUESTION = 300  # 9주차 2교시 «입력 검증» — 지나치게 긴 메시지는 받지 않는다

BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
APP_TOKEN = os.environ.get("SLACK_APP_TOKEN", "")
ALLOWED = {u.strip() for u in os.environ.get("SLACK_ALLOWED_USERS", "").split(",") if u.strip()}

if not BOT_TOKEN.startswith("xoxb-"):
    raise SystemExit("SLACK_BOT_TOKEN 은 xoxb- 로 시작해야 합니다.")
if not APP_TOKEN.startswith("xapp-"):
    raise SystemExit("SLACK_APP_TOKEN 은 xapp- 로 시작해야 합니다.")

app = App(token=BOT_TOKEN)

USAGE = (
    "학칙에 대해 물어봐 주세요 🙂\n"
    "• `@봇 질문` — 학칙에서 찾아 답합니다 (Ollama)\n"
    "• `@봇 openai 질문` — 같은 방식, OpenAI\n"
    "• `@봇 검색 질문` — 어떤 조항이 뽑혔는지만\n"
    "• `@봇 비교 질문` — 학칙 없이 답한 것과 나란히"
)


# ── 1. 메시지 → (모드, 모델, 질문) ───────────────────────────────
def parse(text: str) -> tuple[str, str, str]:
    text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()  # 앞의 <@U07ABCDEF> 를 떼어 낸다
    mode, backend = "rag", DEFAULT_BACKEND
    first, _, rest = text.partition(" ")
    if first.lower() in SEARCH_WORDS:
        return "search", backend, rest.strip()
    if first.lower() in COMPARE_WORDS:
        mode, text = "compare", rest.strip()
        first, _, rest = text.partition(" ")
    if first.lower() in BACKENDS:
        return mode, first.lower(), rest.strip()
    return mode, backend, text


def to_mrkdwn(answer: str) -> str:
    """LLM 은 **굵게** 로 쓰고, 슬랙은 *굵게* 로 읽는다."""
    return answer.strip().replace("**", "*")


def sources_block(docs) -> str:
    """📚 근거 — metadata 가 여기서 쓰인다 ★"""
    return "📚 *근거*  " + "  /  ".join(source_of(d) for d in docs)


def search_text(question: str) -> str:
    lines = [f"🔎 *검색 결과* — 거리(작을수록 가깝다) · `{EMBED_MODEL}`"]
    for i, (doc, dist) in enumerate(search(question), 1):
        preview = doc.page_content[:80].replace("\n", " ")
        lines.append(f"{i}. `{dist:.3f}` *{source_of(doc)}*\n> {preview}…")
    return "\n".join(lines)


# ── 2. 답하기 — 자리부터 잡고, 답이 오면 고친다 (9주차 그대로) ────
def post_holder(client, channel, thread, text):
    return client.chat_postMessage(channel=channel, thread_ts=thread, text=text)


def answer(event: dict, client) -> None:
    channel = event["channel"]
    thread = event.get("thread_ts") or event["ts"]
    user = event.get("user", "")
    mode, backend, question = parse(event.get("text", ""))
    print(f"받음 ← user={user} {mode}/{backend} {question!r}")

    if not question:
        client.chat_postMessage(channel=channel, thread_ts=thread, text=USAGE)
        return
    if len(question) > MAX_QUESTION:
        client.chat_postMessage(
            channel=channel, thread_ts=thread,
            text=f"✋ 질문은 {MAX_QUESTION}자 이내로 부탁드립니다. (지금 {len(question)}자)",
        )
        return
    if backend == "openai" and ALLOWED and user not in ALLOWED:
        client.chat_postMessage(
            channel=channel, thread_ts=thread,
            text="🔒 openai 는 이 봇 주인만 쓸 수 있습니다. 앞에 아무것도 붙이지 않으면 Ollama 가 답합니다.",
        )
        return

    meta = {"slack_user": user, "slack_channel": channel}

    # ── 검색만 — LLM 을 부르지 않으므로 바로 답한다 ──
    if mode == "search":
        try:
            text = search_text(question)
        except Exception as e:
            text = f"⚠️ 검색 실패 — `{type(e).__name__}: {str(e)[:200]}`"
        client.chat_postMessage(channel=channel, thread_ts=thread, text=text)
        return

    # ── 비교 — 먼저 RAG 없이 ──
    if mode == "compare":
        holder = post_holder(client, channel, thread, f"⏳ [RAG 없이] {model_name(backend)} 가 생각 중입니다…")
        try:
            ans, sec = ask_plain(backend, question, meta)
            text = f"🙈 *RAG 없이* (모델 기억만)\n{to_mrkdwn(ans)}\n\n_— {backend} · {model_name(backend)} · {sec:.1f}초_"
        except Exception as e:
            text = f"⚠️ [RAG 없이] 실패 — `{type(e).__name__}: {str(e)[:200]}`"
        client.chat_update(channel=channel, ts=holder["ts"], text=text)

    # ── RAG — ① 자리 잡기 → ② 찾아서 답하기 → ③ 자리를 답으로 ★ ──
    holder = post_holder(client, channel, thread, f"⏳ 학칙에서 찾는 중… ({model_name(backend)})")
    try:
        ans, docs, sec = ask(backend, question, meta)
        text = (
            f"{'📖 *RAG 로* (학칙에서 찾아서)' + chr(10) if mode == 'compare' else ''}"
            f"{to_mrkdwn(ans)}\n\n{sources_block(docs)}\n"
            f"_— rag · {backend} · {model_name(backend)} · {sec:.1f}초_"
        )
        print(f"  답함 → {backend} {sec:.1f}초  근거={[d.metadata.get('article') for d in docs]}")
    except Exception as e:
        text = f"⚠️ {backend} 실패 — `{type(e).__name__}: {str(e)[:200]}`"
        print(f"  실패 → {backend} {type(e).__name__}: {e}")
    client.chat_update(channel=channel, ts=holder["ts"], text=text)


# ── 3. 이벤트 연결 (9주차 그대로) ────────────────────────────────
@app.event("app_mention")
def on_mention(event, client):
    answer(event, client)


@app.event("message")
def on_dm(event, client):
    # (선택) 봇에게 1:1 DM 으로 물을 때 — message.im 이벤트를 구독해야 온다
    if event.get("channel_type") == "im" and not event.get("subtype"):
        answer(event, client)


if __name__ == "__main__":
    print("=" * 60)
    print(f"인덱스   : {INDEX_DIR}")
    print(f"임베딩   : {EMBED_MODEL}")
    print(f"기본 모델 : {DEFAULT_BACKEND} ({model_name(DEFAULT_BACKEND)})")
    print(f"openai 허용 : {', '.join(sorted(ALLOWED)) if ALLOWED else '누구나 ⚠️ 비용 주의'}")

    # ★ 켜기 전에 인덱스를 한 번 읽고 검색해 본다 — 임베딩 모델 워밍업 겸 설정 확인
    try:
        top = search("학칙", k=1)[0][0]
        print(f"인덱스 확인 ✅ ({source_of(top)})")
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(
            f"인덱스 검색 실패 — {type(e).__name__}: {e}\n"
            "  → ollama 가 켜져 있나?  EMBED_MODEL 이 인덱스를 만든 모델과 같은가? (python rag_backend.py 검색)"
        )

    print("슬랙에서 @내앱이름 질문 / 검색 질문 / 비교 질문.  끝낼 때는 Ctrl+C")
    print("=" * 60)
    try:
        SocketModeHandler(app, APP_TOKEN).start()
    except SlackApiError as e:
        raise SystemExit(f"앱 토큰으로 연결하지 못했습니다: {e.response['error']}  → 9주차 실습 1 socket_hello.py 부터 다시")
