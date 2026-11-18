"""[실습 1 덧붙임] 사전 구축 에이전트 — react_agent.py 의 그래프를 한 줄로

  react_agent.py 에서 손으로 조립한 것

      StateGraph(State) + agent 노드 + ToolNode + tools_condition + 되돌아가는 엣지

  을 LangChain 이 함수 하나로 만들어 줍니다.

      from langchain.agents import create_agent
      agent = create_agent(model, tools=TOOLS, system_prompt="...")

  ⚠️ 이름이 바뀌었습니다
     예전 자료의  from langgraph.prebuilt import create_react_agent
     는 LangGraph 1.0 에서 «deprecated» 가 되었고, 지금은 langchain.agents.create_agent 를 씁니다.
     (불러올 수는 있지만 호출하면 경고가 납니다)

  ★ 만들어진 것도 그래프입니다 — invoke / stream / get_graph 를 그대로 씁니다.
    노드 이름만 다릅니다: agent → model, tools → tools

  왜 손으로 먼저 조립했나
     안이 보여야 고칠 수 있기 때문입니다. 리듀서를 빼면 어떻게 되는지(--broken),
     도구 앞에서 멈추려면 어디를 고쳐야 하는지(hitl_agent.py) 는 조립해 본 사람만 압니다.
     (3주차에 ollama.chat() 을 먼저 보고 ChatOllama 로 간 것과 같은 순서)

실행:
    python prebuilt_agent.py
"""

import sys
import warnings

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from react_agent import RECURSION_LIMIT, SYSTEM, TOOL_MODEL, TOOLS, get_tool_llm

agent = create_agent(get_tool_llm(), tools=TOOLS, system_prompt=SYSTEM.content)


def main() -> None:
    print(f"도구 모델: {TOOL_MODEL}\n")

    print("── 만들어진 그래프의 노드 ──")
    print("  ", list(agent.get_graph().nodes))  # ['__start__', 'model', 'tools', '__end__']

    print("\n── 같은 질문으로 실행 ──")
    out = agent.invoke({"messages": [HumanMessage("357 곱하기 4891은?")]},
                       config={"recursion_limit": RECURSION_LIMIT})
    for m in out["messages"]:
        if getattr(m, "tool_calls", None):
            print(f"  [AI]   도구 요청 → {[c['name'] for c in m.tool_calls]}")
        else:
            print(f"  [{m.type}] {str(m.content).strip()[:100]}")

    print("\n── 예전 이름 create_react_agent 를 부르면 ──")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from langgraph.prebuilt import create_react_agent

        create_react_agent(get_tool_llm(), TOOLS)
    for w in caught:
        print(f"  {type(w.message).__name__}: {str(w.message).splitlines()[0][:170]}")

    print("""
관찰 포인트

  노드 이름이 model · tools       → react_agent.py 의 agent · tools 와 같은 구조
  메시지 순서 human → ai(도구 요청) → tool → ai(답)
                                  → react_agent.py 와 같은 순환이 한 번 돌았다
  create_react_agent 경고          → 예전 자료를 볼 때 import 경로를 확인하세요
""")


if __name__ == "__main__":
    main()
