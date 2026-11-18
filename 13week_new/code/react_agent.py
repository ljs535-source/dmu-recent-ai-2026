"""[실습 1] ReAct 에이전트 — 9주차 도구를 그대로, 순환은 그래프가 돌린다

                    ┌──────────────────────────────┐
                    ▼                              │
   START ──▶ ┌────────────┐  도구 필요?  ┌──────────────┐
             │   agent    │─────예──────▶│  ToolNode    │──┘
             │ (모델 호출) │              │ (도구 실행)   │
             └─────┬──────┘              └──────────────┘
                   │ 아니오
                   ▼
                  END

  9주차에 우리가 **손으로** 하던 «도구 실행 → 결과 되돌리기» 를 ToolNode 가 대신합니다.
  그리고 **몇 번 돌지는 모델이 정합니다.** 그게 **에이전트**입니다.

     |            | 9주차              | 13주차                  |
     | 도구 실행   | 우리 코드 (for 문)  | **ToolNode**            |
     | 반복       | 한 바퀴 (수동)      | **모델이 정하는 만큼**    |
     | 종료 판단   | 없음               | **조건부 엣지** (tools_condition) |
     | 상태       | messages 를 손으로  | **State + add_messages** |

  ⚠️ 바뀌지 않은 것: 도구를 **실행하는 주체는 여전히 우리 쪽 코드**입니다.
     ToolNode 는 우리가 그래프에 넣은 노드입니다. 모델이 실행하는 게 아닙니다.
     → 그래서 hitl_agent.py 에서 그 자리를 **멈출 수 있습니다.**

  ⚠️ 도구 호출 모델
     gemma3:4b 는 Ollama 에서 도구 호출을 지원하지 않습니다 («does not support tools»).
     그래서 기본값이 9주차와 같은 llama3.2:3b 입니다.
     루트 .env 에 TOOL_MODEL=gpt-4o-mini 처럼 gpt- 로 시작하는 이름을 적으면 OpenAI 로 돕니다
     (4주차에 넣은 공용 키 OPENAI_API_KEY 로 — 수업 실습용).

  ⚠️ 소형 모델의 실패 3종
     무한 루프           같은 도구를 계속 부른다          → recursion_limit (기본 10)
     도구를 한 번도 안 씀  도구가 있는데 지어낸다           → 도구 2~3개, docstring 개선
     조기 종료           결과를 받고도 답을 안 만든다      → 시스템 프롬프트에 종료 조건

실행:
    python react_agent.py              # 단계별로 찍어 보기 (도구 2개를 쓰는 질문)
    python react_agent.py --broken     # 리듀서를 빼면? → 도구 결과가 사라진다
"""

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()  # ★ 반드시 맨 위 — 저장소 맨 위의 .env 를 찾아 읽습니다

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

TOOL_MODEL = os.getenv("TOOL_MODEL", "llama3.2:3b")  # ⚠️ gemma3:4b 는 도구 호출 미지원
RECURSION_LIMIT = int(os.getenv("WEEK13_RECURSION_LIMIT", "10"))


def get_tool_llm(model: str | None = None):
    """도구 호출용 모델. 이름이 gpt- 로 시작하면 OpenAI, 아니면 Ollama."""
    model = model or TOOL_MODEL
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)


# ── ① 도구 — 9주차 tool_flow.py 의 두 개 + 덧셈 (memory_agent.py 의 후속 질문용) ──
@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다. 정확한 곱셈이 필요할 때 사용한다."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다. 정확한 덧셈이 필요할 때 사용한다."""
    return a + b


@tool
def get_time(timezone: str = "Asia/Seoul") -> str:
    """지정한 시간대의 현재 시각을 문자열로 반환한다."""
    from datetime import datetime
    from zoneinfo import ZoneInfo  # Windows 는 tzdata 패키지가 필요합니다

    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [multiply, add, get_time]  # 소형 모델에는 2~3개가 적당합니다 (9주차와 같은 원칙)

# 조기 종료 · 무한 루프를 막는 두 문장 — 소형 모델에는 이 두 줄이 큽니다.
SYSTEM = SystemMessage(
    "너는 도구를 사용하는 조수다. 도구가 필요하면 호출하고, "
    "도구 결과를 받으면 반드시 최종 답변을 한국어로 작성하라. "  # 조기 종료 방지
    "같은 도구를 같은 인자로 두 번 이상 호출하지 마라.")  # 무한 루프 방지


# ── ② 상태 — 리듀서가 핵심 ──────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]  # ★ 없으면 도구 결과가 덮어써져 사라진다


class BrokenState(TypedDict):
    """리듀서를 일부러 뺀 것 (--broken 실험용)."""

    messages: list  # ← Annotated 제거


def build_agent(tools=None, state_cls=State, model: str | None = None):
    """agent ↔ ToolNode 순환 그래프의 builder 를 만든다.

    memory_agent.py · hitl_agent.py · time_travel.py 가 이 함수를 그대로 가져다 씁니다.
    """
    tools = tools if tools is not None else TOOLS
    llm_with_tools = get_tool_llm(model).bind_tools(tools)

    # ── ③ agent 노드 ────────────────────────────────────────
    def agent(state) -> dict:
        return {"messages": [llm_with_tools.invoke([SYSTEM] + list(state["messages"]))]}

    # ── ④ 조립 ─────────────────────────────────────────────
    builder = StateGraph(state_cls)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))  # ★ 도구 실행을 대신해 준다

    builder.add_edge(START, "agent")
    # tools_condition: «마지막 메시지에 tool_calls 가 있으면 'tools', 없으면 END»
    #   — 12주차에 손으로 짠 라우팅 함수와 같은 일을 합니다.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # ★ 되돌아간다 = 순환

    return builder


builder = build_agent()
graph = builder.compile()


def run(g, question: str) -> dict:
    """노드가 끝날 때마다 무엇을 했는지 찍는다 — 순환이 실제로 도는 것이 보입니다.

    stream_mode="updates" : 노드 하나가 끝날 때마다 «그 노드가 바꾼 것» 만 나옵니다.
    """
    count = {"agent": 0, "tools": 0}
    for step in g.stream({"messages": [HumanMessage(question)]},
                         config={"recursion_limit": RECURSION_LIMIT},
                         stream_mode="updates"):
        for node, update in step.items():
            count[node] = count.get(node, 0) + 1
            for m in (update or {}).get("messages", []):
                if getattr(m, "tool_calls", None):
                    calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
                    print(f"  [{node} #{count[node]}] 도구 요청 → {calls}")
                elif m.type == "tool":
                    print(f"  [{node} #{count[node]}] {m.name} 결과 = {m.content}")
                else:
                    print(f"  [{node} #{count[node]}] 최종 답변 = {m.content.strip()[:120]}")
    print(f"\n  agent 노드 {count['agent']}번 · tools 노드 {count['tools']}번 실행")
    return count


def demo_broken() -> None:
    """리듀서를 빼면 도구 결과(ToolMessage)가 messages 를 덮어써서 사라집니다."""
    from langgraph.errors import GraphRecursionError

    print("""리듀서를 뺀 State 로 돌립니다

    class BrokenState(TypedDict):
        messages: list          # ← Annotated[list, add_messages] 제거
""")
    g = build_agent(state_cls=BrokenState).compile()
    try:
        out = g.invoke({"messages": [HumanMessage("357 곱하기 4891은?")]},
                       config={"recursion_limit": RECURSION_LIMIT})
        msgs = out["messages"]
        print(f"  끝났을 때 messages 개수: {len(msgs)}  (정상 State 는 4개)")
        for m in msgs:
            print(f"   - {m.type}: {str(m.content).strip()[:80].replace(chr(10), ' ')}")
        print("""
  → 질문(Human)이 사라졌습니다. 모델은 앞에서 무슨 일이 있었는지 모른 채
    마지막 메시지 하나만 보고 답합니다. 모델에 따라 같은 도구를 계속 부르다가
    recursion_limit 에 걸리기도 합니다.""")
    except GraphRecursionError:
        print(f"""
  GraphRecursionError — recursion_limit({RECURSION_LIMIT}) 에서 멈췄습니다.
  도구 결과를 못 본 모델이 같은 도구를 계속 요청한 것입니다.
  recursion_limit 이 걸려 있어서 무한히 돌지 않았습니다 (안전망).""")


def main() -> None:
    print(f"도구 모델: {TOOL_MODEL} / recursion_limit {RECURSION_LIMIT}\n")

    if "--broken" in sys.argv:
        demo_broken()
        return

    print("Q1: 357 곱하기 4891은? 그리고 지금 몇 시야?")
    run(graph, "357 곱하기 4891은? 그리고 지금 몇 시야?")

    print("\nQ2: 안녕! 너는 무엇을 할 수 있어?   (도구가 필요 없는 질문)")
    run(graph, "안녕! 너는 무엇을 할 수 있어?")

    print("""
관찰 포인트

  Q1 에서 agent 노드가 두 번 실행됨     → 순환이 돌았다 (agent → tools → agent)
  ToolNode 가 도구를 자동으로 실행      → 9주차에 손으로 짠 for 문이 사라졌다
  도구 요청이 한 번에 두 개일 수 있음    → ToolNode 가 둘 다 실행해 결과 두 개를 돌려준다
  Q2 가 agent 한 번에 끝나면          → 도구가 필요 없어서 바로 END (기대한 경로)
  Q2 에서도 도구를 불렀다면            → 소형 모델의 «필요 없는 도구 호출».
                                        시스템 프롬프트를 답에 그대로 옮겨 적기도 합니다.
  정답 확인: 357 × 4891 = 1,746,087

  모델을 바꿔 비교해 보세요 (루트 .env 의 TOOL_MODEL):
     llama3.2:3b → llama3.2:1b → gpt-4o-mini (공용 키)

  다음: python react_agent.py --broken   (리듀서를 빼면 어떻게 되는가)
""")


if __name__ == "__main__":
    main()
