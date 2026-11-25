"""[1교시 판서 · 3교시 실습 1] ★★ 리듀서 — 덮어쓸까, 누적할까

  리듀서를 **뺐을 때와 넣었을 때**를 나란히 돌려 대화 이력이 사라지는 장면을 봅니다.

  핵심 질문: 두 노드가 **같은 키**를 갱신하면 어떻게 됩니까?

     [리듀서 없음 — 기본 동작은 덮어쓰기]
        A 실행 후: {"messages": ["A가 한 말"]}
        B 실행 후: {"messages": ["B가 한 말"]}     ⚠️ A의 말이 사라졌다!

     [add_messages 를 붙이면 — 누적]
        A 실행 후: {"messages": ["A가 한 말"]}
        B 실행 후: {"messages": ["A가 한 말", "B가 한 말"]}    ✅ 쌓인다

  ⚠️⚠️ 이것이 초심자가 가장 많이 겪는 함정입니다.
       *"대화 이력이 매번 지워지는데 원인을 모르겠다"* 의 정체가 **리듀서 누락**입니다.
       13주차 에이전트에서는 이 누락 때문에 앞 대화와 도구 결과가 사라져 **엉뚱한 답**이 나옵니다. ★★

  | 필드       | 리듀서            | 동작       | 언제                |
  | question   | 없음              | 덮어쓰기   | 값 하나만 유지하면 될 때 |
  | messages   | **add_messages** ★| **누적**   | 대화 이력            |
  | count      | operator.add      | 더하기     | 재시도 횟수 세기 ★    |

실행:
    python reducer_demo.py        # LLM 호출 없음 · 즉시 · 무료 ★
"""

import operator
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ── ① 리듀서 없음 — 덮어쓰기 ⚠️ ──────────────────────────────
class StateNoReducer(TypedDict):
    messages: list
    count: int


# ── ② 리듀서 있음 — 누적 ★ ──────────────────────────────────
class StateWithReducer(TypedDict):
    messages: Annotated[list, add_messages]  # ★ 누적한다
    count: Annotated[int, operator.add]  # ★ 더한다
    question: str  # 리듀서 없음 → 덮어쓰기


def node_a(state) -> dict:
    return {"messages": ["A가 한 말"], "count": 1}


def node_b(state) -> dict:
    return {"messages": ["B가 한 말"], "count": 1}


def build(state_cls):
    b = StateGraph(state_cls)
    b.add_node("A", node_a)
    b.add_node("B", node_b)
    b.add_edge(START, "A")
    b.add_edge("A", "B")
    b.add_edge("B", END)
    return b.compile()


def show(label: str, out: dict) -> None:
    print(f"  [{label}]")
    print(f"    count    = {out.get('count')}")
    msgs = out.get("messages", [])
    print(f"    messages = {len(msgs)}개")
    for m in msgs:
        # add_messages 는 문자열을 HumanMessage 로 바꿔 줍니다 ★
        content = getattr(m, "content", m)
        print(f"      · {content}   ({type(m).__name__})")
    print()


def main() -> None:
    print("같은 그래프(A → B), State 정의만 다릅니다.\n")

    print("── ① 리듀서 없음 ⚠️ ────────────────────────────────")
    show("덮어쓰기", build(StateNoReducer).invoke({"messages": [], "count": 0}))

    print("── ② add_messages / operator.add ★ ─────────────────")
    show("누적", build(StateWithReducer).invoke(
        {"messages": [], "count": 0, "question": "안녕"}))

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
읽어낼 것 ★★

  ① 리듀서 없음   B 가 A 를 **덮어썼습니다.** A 의 말이 사라졌습니다.
                  count 도 1 입니다 (1 + 1 이 아니라 나중 값으로 교체)

  ② 리듀서 있음   messages 는 2개로 **쌓였고**, count 는 2 로 **더해졌습니다.**

  ★ add_messages 는 그냥 append 가 아닙니다.
    메시지 id 가 같으면 **덮어쓰고**, 없으면 **추가**합니다.
    그래서 상태 수정·타임 트래블(13주차 Time Travel)이 가능합니다.

  ★ 문자열을 넣었는데 HumanMessage 로 바뀐 것에 주목하세요.
    add_messages 는 '메시지 리스트' 를 다루는 전용 리듀서입니다.

  ⚠️ 리듀서는 **필드마다** 정합니다.
     위 ②에서 question 은 리듀서가 없으므로 여전히 덮어쓰기입니다.
     "전부 누적" 이 아니라 **"무엇을 누적할지 설계하는 것"** 입니다. ★

  💡 13주차 예고: 에이전트에서 add_messages 를 빼면
     앞 대화와 도구 결과가 사라져 모델이 **엉뚱한 답**을 냅니다. ★★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
