"""[1교시 코드 읽기 · 3교시 실습 1] Supervisor 패턴 — 관리자가 역할별 담당자에게 일을 나눠 준다

  왜 하나로는 안 되는가

     도구 2~3개   →  잘 고른다
     도구 5개     →  가끔 헷갈린다
     도구 10개    →  ⚠️ 못 고른다        ← 9주차에서 본 현상

     엉뚱한 도구 선택 · 도구를 아예 안 씀 · **지시끼리 충돌**
     ("간결히 답하라" + "근거를 상세히 인용하라" + …)

  사람은 어떻게 하는가 → **역할별로 나누고, 관리자가 배분한다**

                    ┌──────────────┐
        질문 ──────▶│  supervisor  │◀──────────────┐
                    └──────┬───────┘               │ 보고를 읽고 다음 담당자 결정
              ┌────────────┼────────────┐          │
              ▼            ▼            ▼          │
        ┌──────────┐ ┌──────────┐ ┌──────────┐     │
        │researcher│ │calculator│ │  writer  │     │
        │ 학칙 검색 │ │  곱셈    │ │ 최종 정리 │     │
        └────┬─────┘ └────┬─────┘ └────┬─────┘     │
             └──── 보고 ───┴───────────│───────────┘
                                       ▼
                                      END   ← writer 가 쓰면 끝 (구조로 종료를 보장)

  ★ 구조는 12주차 **조건부 엣지 + 순환** 그대로입니다. 새 개념이 아니라 조합입니다.
  ★ researcher · calculator 는 각자 compile() 한 **서브그래프**입니다 — 그래프 안에서 그래프를 부릅니다.
       researcher = 검색 → 요약          (11주차 RAG 모양, 도구 호출 없음)
       calculator = 모델 ↔ ToolNode 순환  (13주차 ReAct 모양, 도구 multiply 하나)
    담당자 안쪽 구조는 서로 달라도 됩니다. 관리자는 **보고 한 줄**만 받습니다.
  ★ 종료는 세 겹: writer → END (구조) / MAX_STEPS 에 닿으면 writer 로 (코드) / recursion_limit (프레임워크)
    + 호출 하나하나에도 상한: num_predict (한 번에 만드는 토큰 수)

  ⚖️ 대가: LLM 호출이 늘어납니다. `--single` 로 단일 에이전트와 비교하세요.

  모델 (역할마다 다르게 고를 수 있다는 것도 멀티 에이전트의 장점입니다)
     supervisor · researcher 요약 · writer = MODEL      (기본 gemma3:4b — 도구 호출이 필요 없음)
     calculator                           = TOOL_MODEL (기본 llama3.2:3b — ⚠️ gemma3:4b 는 Ollama 도구 호출 미지원)

실행:
    python supervisor.py            # 멀티 에이전트 (supervisor + 담당자 3)
    python supervisor.py --single   # 같은 질문을 단일 ReAct 에이전트(도구 2개 · TOOL_MODEL 하나)로
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()  # ★ 반드시 맨 위 — 저장소 맨 위의 .env 를 찾아 읽습니다

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL = os.getenv("MODEL", "gemma3:4b")
TOOL_MODEL = os.getenv("TOOL_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
INDEX_DIR = (HERE / os.getenv("WEEK14_INDEX", "../../10week_new/code/index_recursive")).resolve()

MAX_STEPS = int(os.getenv("WEEK14_MAX_STEPS", "6"))  # ① 우리가 정한 종료 조건
RECURSION_LIMIT = int(os.getenv("WEEK14_RECURSION_LIMIT", "15"))  # ② 프레임워크 안전망

# num_predict = 한 번에 만들 최대 토큰 수 (4주차 max_tokens 와 같은 것)
#   ⚠️ 상한이 없으면 소형 모델이 같은 말을 끝없이 반복할 때 멈추지 않습니다.
NUM_PREDICT = int(os.getenv("WEEK14_NUM_PREDICT", "400"))

VERBOSE = True  # 노드 안에서 진행을 찍을지 — streaming.py 는 자기 방식으로 보여 주려고 끈다


def log(text: str) -> None:
    if VERBOSE:
        print(text)


llm = ChatOllama(model=MODEL, temperature=0, num_predict=NUM_PREDICT)
tool_llm = ChatOllama(model=TOOL_MODEL, temperature=0, num_predict=NUM_PREDICT)


# ── 도구 ─────────────────────────────────────────────────────
_store = None


@tool
def search_rules(query: str) -> str:
    """학교 학칙에서 관련 조문을 찾는다. 휴학·복학·제적·성적·졸업 같은 규정을 확인할 때 사용한다."""
    global _store
    from faiss_io import index_exists, load_index  # 한글 경로 대비 (10·11주차와 같은 파일)

    if _store is None:
        if not index_exists(INDEX_DIR):
            return f"[오류] 10주차 인덱스가 없습니다: {INDEX_DIR}"
        from langchain_ollama import OllamaEmbeddings

        _store = load_index(INDEX_DIR, OllamaEmbeddings(model=EMBED_MODEL))
    docs = _store.similarity_search(query, k=2)
    return "\n\n".join(f"[{d.metadata.get('article', '')}]\n{d.page_content}" for d in docs)


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다. 정확한 곱셈이 필요할 때 사용한다."""
    return a * b


# ── 담당자 ① researcher = 검색 → 요약 (11주차 RAG 모양의 서브그래프) ──
class ResearchState(TypedDict):
    question: str
    docs: str
    report: str


def retrieve(state: ResearchState) -> dict:
    return {"docs": search_rules.invoke({"query": state["question"]})}  # 검색은 코드가 한다


def summarize(state: ResearchState) -> dict:
    msg = llm.invoke(
        "아래 학칙 조문만 근거로, 질문에 필요한 사실(숫자 포함)과 조문 이름을 한두 문장으로 보고하라. "
        f"계산은 하지 말라.\n\n[학칙]\n{state['docs']}\n\n[질문]\n{state['question']}")
    return {"report": msg.content.strip()}


rb = StateGraph(ResearchState)
rb.add_node("retrieve", retrieve)
rb.add_node("summarize", summarize)
rb.add_edge(START, "retrieve")
rb.add_edge("retrieve", "summarize")
rb.add_edge("summarize", END)
research_graph = rb.compile()  # ★ compile() 결과 = 그 자체로 완결된 그래프


# ── 담당자 ② calculator = 모델 ↔ ToolNode (13주차 ReAct 모양의 서브그래프) ──
def build_react(tools: list):
    """모델 ↔ ToolNode 순환 그래프를 만들어 compile() 해서 돌려준다."""

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    bound = tool_llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}

    b = StateGraph(AgentState)
    b.add_node("agent", agent)
    b.add_node("tools", ToolNode(tools))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)  # 도구 호출이 있으면 tools, 없으면 END
    b.add_edge("tools", "agent")
    return b.compile()


calc_graph = build_react([multiply])


# ── 관리자 그래프 ────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 질문 + 담당자들의 보고가 쌓인다
    next: str
    steps: int


WORKERS = ("researcher", "calculator", "writer")


class Route(BaseModel):
    """다음 담당자."""

    next: Literal["researcher", "calculator", "writer"] = Field(
        description="학칙 사실이 아직 없으면 researcher, 숫자가 보고됐고 곱셈이 남았으면 calculator, "
                    "필요한 보고가 다 모였으면 writer")


SUPERVISOR_PROMPT = """너는 팀 관리자다. 담당자는 셋이다.
- researcher: 학칙을 검색해 사실을 보고한다
- calculator: 곱셈을 해서 결과를 보고한다
- writer: 보고를 모아 최종 답변을 쓴다 (writer 가 쓰면 일이 끝난다)
규칙: 계산에 쓸 숫자는 학칙에서 나와야 하므로 researcher 가 먼저다. 이미 보고한 담당자에게 같은 일을 다시 시키지 말라.
지금 보고 현황:
{status}
다음에 일할 담당자 하나를 고르라."""


def status_of(state: State) -> str:
    """누가 이미 보고했는지 — LLM 이 대화를 다 읽고 추측하게 두지 않고 코드가 정리해 준다."""
    done = {getattr(m, "name", None) for m in state["messages"][1:]}
    return "\n".join(f"- {w}: {'보고함' if w in done else '아직'}" for w in ("researcher", "calculator"))


def supervisor(state: State) -> dict:
    step = state.get("steps", 0) + 1
    prompt = SUPERVISOR_PROMPT.format(status=status_of(state))
    try:
        nxt = llm.with_structured_output(Route).invoke(
            [SystemMessage(prompt)] + list(state["messages"])).next
    except Exception as e:  # 소형 모델이 형식을 못 지키면 정리하고 끝낸다
        log(f"  [supervisor {step}] ⚠️ 판단 실패 ({type(e).__name__}) → writer")
        nxt = "writer"
    log(f"  [supervisor {step}] → {nxt}")
    return {"next": nxt, "steps": step}


def route(state: State) -> str:
    # 단계 상한에 닿으면 더 헤매지 말고 writer 에게 정리시킨다 (writer → END)
    if state["steps"] >= MAX_STEPS:
        return "writer"
    return state["next"] if state["next"] in WORKERS else "writer"


def report(name: str, text: str) -> dict:
    """담당자 보고 한 줄 — 관리자는 '누가 무엇을 알아냈는지'만 알면 된다."""
    log(f"  [{name}] {text[:80]}")
    return {"messages": [AIMessage(content=f"[{name} 보고] {text}", name=name)]}


def researcher(state: State) -> dict:
    out = research_graph.invoke({"question": state["messages"][0].content})
    return report("researcher", out["report"])


def calculator(state: State) -> dict:
    # 서브그래프 안에서 오간 도구 호출 메시지는 관리자에게 넘기지 않는다
    reports = "\n".join(m.content for m in state["messages"][1:])
    task = f"질문: {state['messages'][0].content}\n보고:\n{reports}\n보고에 나온 숫자로 multiply 도구를 써서 계산하라."
    out = calc_graph.invoke({"messages": [HumanMessage(task)]}, {"recursion_limit": 8})
    return report("calculator", (out["messages"][-1].content or "").strip())


def writer(state: State) -> dict:
    # ⚠️ 대화 목록을 그대로 넘기면 마지막이 AI 메시지(보고)라서 모델이 빈 답을 내기도 한다 (실측)
    #    → 질문과 보고를 사람 메시지 하나로 묶어서 준다
    reports = "\n".join(m.content for m in state["messages"][1:])
    text = llm.invoke(
        "너는 작성 담당이다. 아래 보고만 근거로 질문에 2~3문장으로 답하라. 근거 조문 이름을 함께 적어라.\n\n"
        f"[질문]\n{state['messages'][0].content}\n\n[보고]\n{reports}").content.strip()
    log(f"  [writer] {text[:80]}")
    return {"messages": [AIMessage(content=f"[writer 최종 답변] {text}", name="writer")]}


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("calculator", calculator)
builder.add_node("writer", writer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route, list(WORKERS))
builder.add_edge("researcher", "supervisor")  # 보고하고 관리자에게 복귀 = 순환
builder.add_edge("calculator", "supervisor")
builder.add_edge("writer", END)  # 최종 답변을 쓰면 끝

graph = builder.compile()

QUESTION = "학칙에서 연속 몇 주를 초과해 결석하면 제적되는지 확인하고, 그 주 수에 7을 곱해 며칠인지 알려줘."
INPUTS = {"messages": [HumanMessage(QUESTION)], "steps": 0}
CONFIG = {"recursion_limit": RECURSION_LIMIT}


# ── 비교용: 도구 2개를 한 에이전트(TOOL_MODEL 하나)에 다 주기 ───────
single_graph = build_react([search_rules, multiply])


class CallCounter(BaseCallbackHandler):
    """LLM 이 몇 번 불렸는지 센다 (LangSmith 의 LLM Run 개수와 같은 값)."""

    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.calls += 1


def run_multi() -> None:
    print(f"모델: supervisor·researcher·writer = {MODEL} / calculator = {TOOL_MODEL}")
    print(f"MAX_STEPS {MAX_STEPS} / recursion_limit {RECURSION_LIMIT}\n")
    print(f"Q: {QUESTION}\n")
    counter = CallCounter()
    t0 = time.perf_counter()
    out = graph.invoke(INPUTS, {**CONFIG, "callbacks": [counter]})
    print("\n" + "=" * 60)
    print("최종:", out["messages"][-1].content.strip()[:300])
    print(f"\nsupervisor 판단 {out['steps']}회 / LLM 호출 {counter.calls}회 / {time.perf_counter() - t0:.0f}초")
    print("=" * 60)


def run_single() -> None:
    print(f"모델: 단일 에이전트 = {TOOL_MODEL} (도구 search_rules · multiply 2개)\n")
    print(f"Q: {QUESTION}\n")
    from langgraph.errors import GraphRecursionError

    counter = CallCounter()
    t0 = time.perf_counter()
    try:
        out = single_graph.invoke({"messages": [HumanMessage(QUESTION)]},
                                  {"recursion_limit": RECURSION_LIMIT, "callbacks": [counter]})
    except GraphRecursionError:  # 같은 도구를 끝없이 다시 부르면 여기서 끊긴다
        print(f"⚠️ recursion_limit {RECURSION_LIMIT} 에 걸려 멈췄습니다 — 도구를 계속 다시 불렀습니다.")
        print(f"LLM 호출 {counter.calls}회 / {time.perf_counter() - t0:.0f}초")
        return
    for m in out["messages"][1:]:
        calls = getattr(m, "tool_calls", None)
        if calls:
            print("  [agent] 도구 호출 →", ", ".join(f"{c['name']}({c['args']})" for c in calls))
        elif m.type == "tool":
            print(f"  [tools] {m.name} → {str(m.content)[:60]!r}")
    print("\n" + "=" * 60)
    print("최종:", (out["messages"][-1].content or "").strip()[:300])
    print(f"\nLLM 호출 {counter.calls}회 / {time.perf_counter() - t0:.0f}초")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_single() if "--single" in sys.argv else run_multi()
