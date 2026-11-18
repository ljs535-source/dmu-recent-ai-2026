"""[실습 2] thread_id 로 대화를 이어 간다 — 1주차 ✗ «어제 얘기를 기억 못 한다» 를 지우는 자리

  지금은 매번 처음부터입니다.

     graph.invoke({"messages": [HumanMessage("357 곱하기 4891은?")]})
     # → "1746087입니다"
     graph.invoke({"messages": [HumanMessage("거기에 2를 더하면?")]})
     # → "무엇에 2를 더하나요?"   ⚠️ 앞 대화를 모른다

  react_agent.py 의 add_messages 는 **한 번의 invoke 안에서** 메시지를 쌓습니다.
  invoke 가 끝나면 상태가 사라집니다.

  해결은 두 가지가 **함께** 있어야 합니다

     | 체크포인터  | 상태를 **저장하는 장치** (어디에 저장할지)          |
     | thread_id  | **어느 대화인지** 식별 — 카카오톡 채팅방 하나와 같다 |

  ★ 이 실습의 핵심은 3번 호출입니다.
    "메모리가 있다" 가 아니라 **"thread 단위로 있다"** 는 것.
    실제 서비스에서 **사용자마다 다른 thread_id** 를 주는 이유입니다.
    안 그러면 남의 대화가 섞입니다 (개인정보 사고).
    슬랙 봇이라면 «채널 + 스레드 ts» 를 thread_id 로 쓰면 스레드마다 대화가 이어집니다.

  | 구분 | 무엇                      | 저장소                          | 이번 실습 |
  | 단기 | 한 대화(thread) 안         | InMemorySaver (프로세스 메모리)  | ✅        |
  | 장기 | 대화를 넘어서 남는 정보     | SQLite · DB 등 영속 저장소        | 개념만    |

  ⚠️ InMemorySaver 는 **프로그램이 끝나면 사라집니다.**
     (예전 자료의 MemorySaver 는 같은 클래스의 옛 이름입니다)

실행:
    python memory_agent.py
"""

import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from react_agent import RECURSION_LIMIT, TOOL_MODEL, builder  # 실습 1 그래프를 그대로

graph = builder.compile(checkpointer=InMemorySaver())  # ★ 이 한 줄


def ask(text: str, thread: str) -> None:
    out = graph.invoke(
        {"messages": [HumanMessage(text)]},
        config={"configurable": {"thread_id": thread},  # ★ 대화 식별자
                "recursion_limit": RECURSION_LIMIT},
    )
    # 이번 질문 뒤에 새로 생긴 메시지에서 도구 요청만 골라 본다
    msgs = out["messages"]
    last_human = max(i for i, m in enumerate(msgs) if m.type == "human")
    calls = [f"{c['name']}({c['args']})"
             for m in msgs[last_human:] for c in (getattr(m, "tool_calls", None) or [])]
    print(f"[{thread}] Q: {text}")
    print(f"[{thread}]    도구 요청: {', '.join(calls) or '없음'}")
    print(f"[{thread}] A: {msgs[-1].content.strip()[:100]}\n")


def main() -> None:
    print(f"도구 모델: {TOOL_MODEL} / 체크포인터: InMemorySaver (끝나면 사라집니다)\n")

    # ── 대화 A ─────────────────────────────────────────────
    ask("357 곱하기 4891은?", "chat-A")
    ask("거기에 2를 더하면?", "chat-A")  # 이어진다

    # ── 대화 B — 다른 thread ────────────────────────────────
    ask("거기에 2를 더하면?", "chat-B")  # 기억이 없다

    # ── 저장된 상태 들여다보기 ──────────────────────────────
    print("── 저장된 상태 ──────────────────────────────────")
    for thread in ("chat-A", "chat-B"):
        state = graph.get_state({"configurable": {"thread_id": thread}})
        print(f"  [{thread}] 메시지 수: {len(state.values['messages'])}  "
              f"다음 노드: {state.next}")  # hitl_agent.py 에서 이 칸이 채워집니다

    print("""
결과 확인표 — 직접 채워 보세요

  | 호출                      | thread     | 앞 대화를 아는가 |
  | 1. "357 곱하기 4891은?"    | chat-A     | —              |
  | 2. "거기에 2를 더하면?"     | chat-A     |                |
  | 3. "거기에 2를 더하면?"     | chat-B     |                |

  3번에서 모델이 «무엇에?» 라고 되묻지 않고 아무 숫자로 답을 지어낼 수도 있습니다.
  그래도 결론은 같습니다 — chat-B 에는 1746087 이 없습니다.

  ⚠️ 2번의 «도구 요청» 줄도 꼭 보세요. 기억은 했는데(답에 1746087 이 나옴)
     도구 인자를 add(2, 2) 처럼 잘못 채우면 답이 틀립니다. 기억과 도구 사용은 별개입니다.

💡 대화가 길어지면?
   매 호출에 전체 이력이 프롬프트로 들어가 토큰이 계속 늘어납니다.
   실무에서는 오래된 메시지를 요약하거나 잘라냅니다
   (11주차 Lost in the Middle 과 같은 이유 — 길다고 좋은 게 아닙니다).

💡 프로그램을 다시 켜도 기억하게 하려면 영속 저장소가 필요합니다.
   pip install langgraph-checkpoint-sqlite
   from langgraph.checkpoint.sqlite import SqliteSaver
""")


if __name__ == "__main__":
    main()
