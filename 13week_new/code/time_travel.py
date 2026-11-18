"""[2교시 코드 읽기 · 3교시 선택] 상태 되감기 (Time Travel) — 체크포인트가 있으니 가능합니다

   실행 이력:  START → agent → tools → agent → END
                 ▼      ▼       ▼       ▼      ▼
               [cp4]  [cp3]   [cp2]   [cp1]  [cp0]     ← 단계마다 저장돼 있다
                              ▲
                       여기로 되감아 다시 실행할 수 있다

  6주차 추적과 비교해 보세요.

     |        | LangSmith 추적 (6주)       | Time Travel (13주)            |
     | 무엇   | 무슨 일이 있었는지 **본다**  | 그 시점으로 돌아가 **다시 한다** |
     | 성격   | 읽기 전용                   | 실행 가능                      |

  LLM 앱은 비결정적이라(6주차) 같은 입력을 다시 넣어도 재현이 안 될 수 있습니다.
  그런데 체크포인트에서 재개하면 **그 지점까지의 상태는 그대로**입니다.

  무엇에 쓰나
     디버깅       «그 단계에서 다른 선택을 했다면?» 을 실제로 돌려 본다
     What-if     상태를 고쳐서 다시 실행 (예: 검색 결과를 바꿔 넣어 보기)
     오류 복구    실패 지점 직전으로 되감아 재시도
     HITL 확장   승인 거부 후 다른 인자로 재시도

실행:
    python time_travel.py
"""

import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from react_agent import RECURSION_LIMIT, TOOL_MODEL, builder

graph = builder.compile(checkpointer=InMemorySaver())
CONFIG = {"configurable": {"thread_id": "tt-1"}, "recursion_limit": RECURSION_LIMIT}


def main() -> None:
    print(f"도구 모델: {TOOL_MODEL}\n")
    print("── ① 먼저 한 번 실행합니다 ──────────────────────\n")
    out = graph.invoke({"messages": [HumanMessage("357 곱하기 4891은?")]}, CONFIG)
    print("최종 답변:", out["messages"][-1].content.strip()[:80])

    # ── ② 지금까지의 체크포인트 목록 ────────────────────────
    print("\n── ② 체크포인트 목록 (최근 → 과거) ─────────────")
    history = list(graph.get_state_history(CONFIG))
    for i, s in enumerate(history):
        cid = s.config["configurable"].get("checkpoint_id", "?")
        last = s.values.get("messages", [])
        tail = getattr(last[-1], "content", "")[:30].replace("\n", " ") if last else ""
        print(f"  [{i}] …{cid[-8:]}  다음: {str(s.next):16s} 메시지 {len(last)}개  {tail}")

    print(f"\n  총 {len(history)}개의 체크포인트가 남아 있습니다 — 되감기의 재료입니다.\n")

    # ── ③ 특정 시점으로 되감아 재실행 ───────────────────────
    # 아직 실행할 노드가 남아 있는(=next 가 비어 있지 않은) 지점을 고릅니다.
    resumable = [s for s in history if s.next]
    if not resumable:
        print("되감을 지점이 없습니다 (한 바퀴에 끝났습니다). 도구를 쓰는 질문으로 바꿔 보세요.")
        return

    past = resumable[len(resumable) // 2]  # 중간쯤의 체크포인트
    print("── ③ 그 지점부터 다시 실행합니다 ────────────────")
    print(f"  되감을 지점: 다음 노드 = {past.next}, 메시지 {len(past.values['messages'])}개\n")

    # ★ invoke(None, past.config) — 입력을 주지 않으면 '그 지점부터 이어서' 실행합니다
    again = graph.invoke(None, past.config)
    print("재실행 결과:", again["messages"][-1].content.strip()[:80])
    print(f"재실행 뒤 체크포인트 수: {len(list(graph.get_state_history(CONFIG)))}개"
          "  (지운 것이 아니라 갈래가 하나 더 생겼습니다)")

    print("""
읽어낼 것

  체크포인트가 단계마다 남아 있다       → 되감기의 재료
  invoke(None, past.config)           → 그 지점부터 이어서 실행
  되감기 뒤 체크포인트가 늘어난다        → 과거를 지우지 않고 새 갈래를 만든다

  6주차 추적은 «본다»(읽기 전용), 되감기는 «다시 한다»(실행 가능).

💡 What-if: graph.update_state(past.config, {...}) 로 상태를 고친 뒤 이어서 실행하면
   «그 단계에서 다른 값이었다면?» 을 실제로 확인할 수 있습니다.
   (14주차 LangGraph Studio 에서는 이것을 화면에서 합니다)
""")


if __name__ == "__main__":
    main()
