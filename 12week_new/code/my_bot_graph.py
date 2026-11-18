"""[3교시 / 실습 5 · 미니 프로젝트] ✏️ 내 슬랙 봇의 분기 그래프 — 고칠 곳만 고쳐서 쓰는 출발 파일

rag_router_graph.py 는 «학칙 / 일반» 두 갈래였습니다.
이 파일은 같은 골격을 **내 프로젝트의 갈래**로 바꾸는 연습용입니다. 예시로 세 갈래를 넣어 두었습니다.

                    ┌────────────┐
      START ───────▶│  classify  │  ← ROUTES 표로 분류 프롬프트를 만든다
                    └─────┬──────┘
          "학칙"          │ "도움말"          "일반"
       ┌──────────────────┼──────────────────────┐
       ▼                  ▼                      ▼
  ┌──────────┐      ┌──────────┐           ┌──────────┐
  │ retrieve │      │   help   │           │ general  │
  └────┬─────┘      └────┬─────┘           └────┬─────┘
       ▼                 │  ← LLM 없이 정해진 글  │  ← 검색 없이 답
  ┌────────────┐         │                      │
  │ rag_answer │         │                      │
  └────┬───────┘         │                      │
       └─────────────────┴──────────┬───────────┘
                                    ▼
                                   END

  ✏️ 고칠 곳은 네 군데입니다 (파일 안에서 «✏️» 를 찾으세요)
     ① ROUTES    갈래 이름 · 기준 · 예시 질문     ← 가장 먼저, 가장 많이 고칠 곳
     ② FALLBACK  분류가 실패했을 때 보낼 갈래      ← «틀렸을 때 덜 위험한 쪽»
     ③ 노드      갈래마다 할 일 (검색 / 정해진 글 / 바로 답 / MCP 도구 …)
     ④ 연결      add_node · add_conditional_edges 표 — 갈래 이름과 노드 이름을 잇는다

  ★ 학칙 경로의 검색 · 근거 답은 rag_router_graph.py 의 retrieve · rag_answer 를 그대로 가져다 씁니다.
    내 문서 인덱스를 쓰려면 루트 .env 의 WEEK12_INDEX 만 바꾸면 됩니다 (rag_router_graph.py 위치 기준 경로).
  ★ MCP 팀은 retrieve 자리에 MCP 도구를 부르는 노드를 넣으면 됩니다 — 그래프 모양은 같습니다.

실행:
    python my_bot_graph.py --route-only                  # ★ 분류만 (인덱스 없이도 됨 · 빠름) — 먼저 이것부터
    python my_bot_graph.py --route-only "내 질문"         # 내 질문 하나만 분류
    python my_bot_graph.py                               # 질문 전부 끝까지 (학칙 갈래는 10주차 인덱스 필요)
    python visualize.py my_bot_graph                     # 그래프 그림 (README 에 붙일 Mermaid)
"""

from dotenv import load_dotenv

load_dotenv()  # ★ 반드시 맨 위

import sys  # noqa: E402
from typing import Literal, TypedDict, get_args  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# 학칙 갈래는 이미 만든 것을 그대로 — rag_router_graph.py · faiss_io.py 가 같은 폴더에 있어야 합니다
from rag_router_graph import MODEL, llm, rag_answer, retrieve  # noqa: E402

# ── ✏️ ① 내 봇의 갈래 — 이름: (기준, [예시 질문 …]) ─────────────────
#    예시 질문은 갈래마다 2~3개. 헷갈리는 질문(구어체 · 경계에 있는 질문)을 넣을수록 분류가 좋아집니다.
ROUTES = {
    "학칙": ("우리 학교 학칙(휴학·복학·자퇴·제적·성적·졸업 등 학사 규정)을 찾아봐야 답할 수 있는 질문",
             ["휴학은 몇 년까지 돼?", "몇 주 결석하면 제적이야?", "학교 그만두려면 어떻게 해?"]),
    "도움말": ("이 봇이 무엇을 할 수 있는지, 어떻게 쓰는지 묻는 질문",
               ["너 뭐 할 수 있어?", "사용법 알려줘"]),
    "일반": ("인사 · 잡담 · 일반 상식 · 프로그래밍 같은 그 밖의 질문",
             ["안녕하세요", "파이썬 리스트 정렬 방법", "LangGraph 가 뭐야?"]),
}

# ── ✏️ ② 분류 실패 시 보낼 갈래 — 학칙 질문을 일반으로 보내면 지어내므로 '학칙' 쪽이 안전합니다
FALLBACK = "학칙"

# ✏️ ① 과 같은 이름을 여기에도 적습니다 (구조화 출력이 이 이름들만 고르게 막아 줍니다)
RouteName = Literal["학칙", "도움말", "일반"]

if set(get_args(RouteName)) != set(ROUTES) or FALLBACK not in ROUTES:
    sys.exit("⚠️ ROUTES · RouteName · FALLBACK 의 갈래 이름이 서로 다릅니다. 세 곳을 똑같이 맞추세요.")


# ── 상태 — rag_router_graph.py 와 같은 키 (retrieve · rag_answer 를 그대로 쓰려고) ──
class State(TypedDict):
    question: str
    route: str
    docs: list[Document]
    answer: str
    sources: list[str]


# ── 분류 노드 — ROUTES 표에서 프롬프트를 만든다 (고칠 필요 없음) ─────────
class Route(BaseModel):
    """질문을 어느 갈래로 보낼지."""

    route: RouteName = Field(description=" / ".join(f"{k}: {v[0]}" for k, v in ROUTES.items()))


def build_classify_prompt() -> ChatPromptTemplate:
    rules = "\n".join(f"- '{k}': {crit}" for k, (crit, _) in ROUTES.items())
    examples = " / ".join(f"'{q}' → {k}" for k, (_, qs) in ROUTES.items() for q in qs)
    return ChatPromptTemplate.from_messages([
        ("system",
         "너는 대학 슬랙 봇의 질문 분류기다. 질문을 아래 갈래 중 하나로 분류한다.\n"
         f"{rules}\n예시) {examples}"),
        ("human", "{q}"),
    ])


CLASSIFY_PROMPT = build_classify_prompt()


def classify(state: State) -> dict:
    try:
        route = (CLASSIFY_PROMPT | llm.with_structured_output(Route)).invoke({"q": state["question"]}).route
    except Exception as e:
        print(f"  [분류] ⚠️ 실패 ({type(e).__name__}) → {FALLBACK} 갈래로 보냅니다")
        route = FALLBACK
    print(f"  [분류] {route}")
    return {"route": route}


# ── ✏️ ③ 갈래마다 할 일 ──────────────────────────────────────────
#    학칙 → retrieve · rag_answer (rag_router_graph.py 에서 가져옴)
HELP_TEXT = ("저는 학칙 안내 봇입니다. 휴학·복학·제적·성적·졸업 같은 학칙을 물어보면 근거 조항과 함께 답합니다. "
             "그 밖의 질문은 검색 없이 짧게 답합니다.")


def help_node(state: State) -> dict:
    """LLM 을 부르지 않는 노드 — 정해진 글이면 비용 0, 답도 항상 같습니다."""
    return {"answer": HELP_TEXT, "sources": []}


GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 대학 수업 슬랙 채널의 조교 봇이다. 한국어로 세 문장 이내로 답한다. "
               "학교 규정 · 일정 · 시간처럼 확인이 필요한 사실은 추측하지 말고 모른다고 말한다."),
    ("human", "{question}"),
])


def general(state: State) -> dict:
    ans = (GENERAL_PROMPT | llm | StrOutputParser()).invoke({"question": state["question"]})
    return {"answer": ans.strip(), "sources": []}


# ── 라우팅 함수 — 갈래 이름 → 첫 노드 이름 (읽기만 한다) ─────────────
FIRST_NODE = {"학칙": "retrieve", "도움말": "help", "일반": "general"}  # ✏️ ④ 갈래를 늘리면 여기도


def route_after_classify(state: State) -> str:
    return FIRST_NODE.get(state["route"], FIRST_NODE[FALLBACK])


# ── ✏️ ④ 조립 ───────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("rag_answer", rag_answer)
builder.add_node("help", help_node)
builder.add_node("general", general)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_after_classify,
                              {name: name for name in FIRST_NODE.values()})
builder.add_edge("retrieve", "rag_answer")
builder.add_edge("rag_answer", END)
builder.add_edge("help", END)
builder.add_edge("general", END)

graph = builder.compile()


def answer(question: str) -> str:
    """★ 슬랙 봇이 부를 함수 — rag_router_graph.answer() 와 쓰는 법이 같습니다."""
    out = graph.invoke({"question": question, "docs": [], "sources": []})
    tail = f"\n\n📎 근거: {', '.join(out['sources'])}" if out["sources"] else ""
    return out["answer"] + tail


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # ✏️ 내 봇에 실제로 들어올 질문으로 바꾸세요 — 갈래마다 하나 이상 + 헷갈리는 것 하나
    questions = args or [
        "졸업하려면 몇 학점 필요해?",
        "이 봇은 뭘 물어보면 돼?",
        "오늘 점심 뭐 먹지",
        "수업 빠지면 어떻게 돼?",
    ]
    print(f"모델: {MODEL} / 갈래: {' · '.join(ROUTES)} / 실패 시: {FALLBACK}\n")

    if "--route-only" in sys.argv:
        for q in questions:
            print(f"Q: {q}")
            classify({"question": q})
        print("\n→ 기록표에 적으세요: 질문 · 내가 생각한 갈래 · 분류 결과 · 맞음/틀림")
        return

    for q in questions:
        print(f"\nQ: {q}")
        print("A:", answer(q).replace("\n", "\n   "))


if __name__ == "__main__":
    main()
