"""[2교시 · 3교시 실습 3] Studio 에서 승인 버튼으로 다시 해 보는 13주차 Human-in-the-Loop

  13주차에는 터미널에서 approve / reject 를 입력했습니다.
  같은 그래프를 `langgraph dev` 로 띄우면 Studio 화면에서 **멈춘 지점과 승인 요청**을 보고
  값을 넣어 재개할 수 있습니다. 코드는 같고, 사용자 경험만 달라집니다.

     [agent] ── delete_file 호출 ──▶ [tools]
                                      │ interrupt — 여기서 멈춤 ⏸
                                      ▼
                         Studio: resume 값 "approve" / "reject"

  ⚠️ 이 파일의 `graph` 는 **체크포인터 없이** compile() 합니다.
     `langgraph dev` 서버가 자기 저장소(체크포인터)를 붙이기 때문입니다.
     직접 compile(checkpointer=MemorySaver()) 해 두면 langgraph dev 가 거부합니다.
     터미널에서 돌릴 때(아래 main)만 MemorySaver 를 붙입니다.

  ⚠️ interrupt 앞에는 부작용(파일 쓰기·메일 발송 등)을 두지 않습니다.
     재개하면 그 노드가 **처음부터 다시** 실행되기 때문입니다 (13주차 내용).

실행:
    python studio_hitl.py --auto approve   # 터미널에서 확인 (자동 승인)
    python studio_hitl.py --auto reject    # 터미널에서 확인 (자동 거부)
    langgraph dev                          # Studio 에서 "hitl" 그래프 선택
"""

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

TOOL_MODEL = os.getenv("TOOL_MODEL", "llama3.2:3b")  # ⚠️ gemma3:4b 는 도구 호출 미지원


@tool
def list_files() -> list[str]:
    """작업 폴더의 파일 목록을 조회한다."""
    return ["report.pdf", "notes.txt", "data.csv"]


@tool
def delete_file(filename: str) -> str:
    """작업 폴더의 파일 하나를 삭제한다. 되돌릴 수 없다."""
    decision = interrupt({  # ⏸ 여기서 멈춘다 — 앞에는 부작용 없음
        "action": "delete_file",
        "filename": filename,
        "message": f"'{filename}' 을(를) 삭제하려 합니다. approve 또는 reject 를 넣어 주세요.",
    })
    if decision != "approve":
        return f"사용자가 거부하여 '{filename}' 을(를) 삭제하지 않았습니다."
    return f"[시뮬레이션] '{filename}' 삭제됨"  # 실제로 지우지 않습니다


TOOLS = [list_files, delete_file]
bound = ChatOllama(model=TOOL_MODEL, temperature=0, num_predict=400).bind_tools(TOOLS)  # 출력 상한 (supervisor.py 참고)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: State) -> dict:
    sys_msg = SystemMessage("너는 파일 관리 도우미다. 필요한 도구를 쓰고, 도구 결과를 받으면 한국어로 짧게 답하라. "
                            "사용자가 거부한 작업은 다시 시도하지 말고 거부되었다고 알려라.")
    return {"messages": [bound.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()  # ★ 체크포인터 없이 — langgraph dev 가 붙인다


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from langgraph.checkpoint.memory import MemorySaver

    local = builder.compile(checkpointer=MemorySaver())  # 터미널용
    answer = sys.argv[sys.argv.index("--auto") + 1] if "--auto" in sys.argv else None
    config = {"configurable": {"thread_id": "studio-hitl-1"}, "recursion_limit": 10}

    print(f"모델: {TOOL_MODEL}\nQ: report.pdf 를 삭제해줘\n")
    local.invoke({"messages": [HumanMessage("report.pdf 를 삭제해줘")]}, config)
    # 멈출 때마다 사람이 판단한다 — 거부해도 모델이 다시 시도하면 또 멈춘다 (최대 3번)
    for _ in range(3):
        state = local.get_state(config)
        if not state.next:
            break
        print("⏸ 멈춤 — 다음 노드:", state.next)
        for task in state.tasks:
            for intr in task.interrupts:
                print("   승인 요청:", intr.value)
        decision = answer or input("approve / reject ? ").strip()
        print(f"   사람의 판단: {decision}\n")
        local.invoke(Command(resume=decision), config)

    state = local.get_state(config)
    if state.next:
        print("⚠️ 3번 거부했는데도 모델이 계속 같은 도구를 부르려 합니다 — 그래도 실행되지는 않았습니다.")
    for m in state.values["messages"][1:]:
        calls = getattr(m, "tool_calls", None)
        if calls:
            print("  [agent] 도구 호출 →", ", ".join(f"{c['name']}({c['args']})" for c in calls))
        elif m.type == "tool":
            print(f"  [tools] {m.content}")
        else:
            print(f"  [agent] {m.content}")


if __name__ == "__main__":
    main()
