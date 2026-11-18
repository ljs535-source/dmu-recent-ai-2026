"""[2교시 / 판서 · 3교시] ★ 종료 조건을 빼면 — recursion_limit 이 그래프를 멈춘다

   START ─▶ [work] ─▶ [check] ──"다시"──┐
               ▲                         │
               └─────────────────────────┘      ← 끝나는 길이 없다

  check 가 언제나 "다시" 를 돌려주면 그래프는 영원히 돕니다.
  LangGraph 는 한 번 실행에서 거친 **단계(super-step) 수**를 세다가
  recursion_limit 에 닿으면 GraphRecursionError 를 내고 멈춥니다.

  ⚠️ 기본값이 생각보다 큽니다 — langgraph 1.2 의 기본 한도는 **10007** 단계입니다
     (예전 판과 인터넷 자료에는 25 로 나옵니다). LLM 노드라면 사실상 '무한' 입니다.
     그래서 config={"recursion_limit": 10} 처럼 **직접 낮게 적어 주어야** 합니다 ★

  ★ 이것은 '정상 종료' 가 아니라 '사고 차단' 입니다.
    정상 종료는 상태(attempts 같은 값)를 보고 우리가 END 로 보내도록 설계합니다 (reflection.py).
    LLM 을 부르는 노드였다면 한도에 닿기 전에 몇 시간이 지나갑니다 ⚠️

실행:
    python recursion_demo.py           # LLM 호출 없음 · 즉시 · 무료
"""

import operator
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402


class State(TypedDict):
    attempts: Annotated[int, operator.add]  # ★ 리듀서로 1씩 더한다


def work(state: State) -> dict:
    return {"attempts": 1}


def check_forever(state: State) -> str:
    return "다시"  # ⚠️ 종료 조건이 없다


def check_with_limit(state: State) -> str:
    return "끝" if state["attempts"] >= 3 else "다시"  # ★ 논리적 종료


def build(router):
    b = StateGraph(State)
    b.add_node("work", work)
    b.add_edge(START, "work")
    b.add_conditional_edges("work", router, {"다시": "work", "끝": END})
    return b.compile()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("── ① 아무것도 적지 않았을 때의 한도 ─────────────────")
    try:
        from langgraph._internal._config import DEFAULT_RECURSION_LIMIT

        print(f"  langgraph 기본 recursion_limit = {DEFAULT_RECURSION_LIMIT}   ⚠️ 1만 단계 — 직접 낮춰야 합니다")
    except ImportError:
        print("  (이 버전에서는 기본값 위치가 다릅니다 — 공식 문서를 확인하세요)")

    print("\n── ② 종료 조건 없음 + recursion_limit=10 ────────────")
    try:
        build(check_forever).invoke({"attempts": 0}, config={"recursion_limit": 10})
    except GraphRecursionError as e:
        print(f"  {type(e).__name__}:")
        print("  " + str(e).splitlines()[0])

    print("\n── ③ 종료 조건 있음 (attempts >= 3 이면 END) + recursion_limit=10 ──")
    out = build(check_with_limit).invoke({"attempts": 0}, config={"recursion_limit": 10})
    print(f"  정상 종료 — attempts = {out['attempts']}")

    print("""
읽어낼 것 ★

  ①  아무것도 안 적으면 한도가 1만 단계       LLM 노드였다면 호출 1만 번 — 사실상 무한 ⚠️
  ②  한도를 10 으로 적으면 10 단계에서 멈춘다  실습에서는 한도를 낮게 (8~10)
  ③  종료 조건이 있으면 오류 없이 끝난다       정상 종료는 우리가 설계한다 ★★
""")


if __name__ == "__main__":
    main()
