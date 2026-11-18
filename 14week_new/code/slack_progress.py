"""[3교시 · 미니 프로젝트 연결] 상태 단위 스트리밍을 슬랙 메시지 하나로 보여 준다

  9·10주차 슬랙 봇은 이미 이렇게 합니다.

     ① chat_postMessage  "⏳ 학칙에서 찾는 중…"     ← 자리부터 잡는다
     ② (오래 걸리는 처리)
     ③ chat_update       같은 메시지를 답으로 고친다

  오늘은 ②를 **그래프의 stream_mode="updates"** 로 돌리면서,
  노드 하나가 끝날 때마다 ③을 여러 번 부릅니다.

     ⏳ 담당자를 정하는 중…
     ⏳ 학칙을 찾는 중…  (✔ 담당자 정함)
     ⏳ 계산하는 중…     (✔ 학칙 찾음)
     …
     ✅ 최종 답변

  이 파일은 **슬랙에 접속하지 않습니다.** 슬랙 client 와 같은 메서드 이름을 가진
  가짜 client(FakeClient)로 어떤 순서로 메시지가 바뀌는지만 보여 줍니다.
  내 봇에 붙일 때는 `answer_with_progress(event, client)` 를 10주차
  `slack_rag_bot.py` 의 answer() 자리에 쓰면 됩니다 (README «미니 프로젝트로 이어 가기»).

  ⚠️ 슬랙 chat_update 는 너무 자주 부르면 rate limit(초당 약 1회) 에 걸립니다.
     노드 단위(수 초에 한 번)는 괜찮지만 **토큰 단위로 고치면 안 됩니다.**

실행:
    python slack_progress.py
"""

import sys
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from streaming import LABELS  # (streaming 을 불러오면 supervisor 노드 안의 print 도 꺼집니다)
from supervisor import CONFIG, graph

DONE = {"supervisor": "배정", "researcher": "학칙 검색", "calculator": "계산", "writer": "정리"}  # 끝난 단계 이름표


def answer_with_progress(event: dict, client) -> None:
    """슬랙 이벤트 하나를 받아, 진행 상황을 같은 메시지에 고쳐 쓰며 답한다."""
    channel = event["channel"]
    thread = event.get("thread_ts") or event["ts"]
    question = event.get("text", "").strip()

    holder = client.chat_postMessage(channel=channel, thread_ts=thread, text="⏳ 시작합니다…")
    done: list[str] = []
    final = ""
    for chunk in graph.stream({"messages": [HumanMessage(question)], "steps": 0},
                              CONFIG, stream_mode="updates"):
        for node, update in chunk.items():
            done.append(DONE.get(node, node))
            if node == "writer":
                final = update["messages"][-1].content.replace("[writer 최종 답변] ", "")
            elif node == "supervisor" and update.get("next") in LABELS:
                text = f"⏳ {LABELS[update['next']]}…\n_({' → '.join(done)})_"
                client.chat_update(channel=channel, ts=holder["ts"], text=text)

    client.chat_update(channel=channel, ts=holder["ts"],
                       text=f"✅ {final or '답을 만들지 못했습니다.'}\n_({' → '.join(done)})_")


class FakeClient:
    """slack_sdk WebClient 흉내 — 받은 호출을 화면에 찍기만 한다."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def _log(self, kind: str, text: str) -> None:
        print(f"[{time.perf_counter() - self.t0:5.1f}s] {kind:<16} {text}")

    def chat_postMessage(self, channel, thread_ts, text):
        self._log("chat_postMessage", text)
        return {"ts": "1700000000.000100"}

    def chat_update(self, channel, ts, text):
        self._log("chat_update", text.replace("\n", "  "))
        return {"ok": True}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from supervisor import QUESTION

    event = {"channel": "C0TEST", "ts": "1700000000.000001", "text": QUESTION}
    print(f"Q: {QUESTION}\n")
    answer_with_progress(event, FakeClient())
