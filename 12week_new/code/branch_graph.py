"""[2교시 판서 · 3교시 실습 2] ★★ 조건부 엣지 — 질문 유형을 분류해 서로 다른 경로로 보낸다

                    ┌─────────────┐
      START ───────▶│  classify   │
                    └──────┬──────┘
                           │  ★ 라우팅 함수가 '문자열' 을 돌려준다
              ┌────────────┼────────────┐
       "math" │      "rag" │    "chat"  │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ calc     │ │ search   │ │ chat     │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └────────────┼────────────┘
                          ▼
                         END

  ★ 라우팅 함수는 노드가 아닙니다.
    상태를 **읽기만** 하고 **다음 목적지 이름**을 돌려줍니다.
    여기서 LLM 을 부르지 않습니다 — 분류는 앞 노드(classify)가 이미 했습니다.

  ★ Literal 타입으로 분류 결과를 제한한 것에 주목하세요.
    5주차 구조화 출력이 여기서 값을 합니다 —
    모델이 "수학문제" 같은 엉뚱한 문자열을 돌려주면 라우팅 딕셔너리에 없어서 그래프가 멈춥니다.
    스키마가 그것을 막습니다.

  ★ 분류 프롬프트 두 가지를 비교해 보세요 (--plain)
     --plain   "질문을 분류하라: {q}" 한 줄            ← 처음 떠올리기 쉬운 프롬프트
     (기본)    유형 설명 + 예시를 넣은 시스템 프롬프트    ← 분류기도 프롬프트 설계가 필요하다

  📌 과제 5 가 이 실습의 확장입니다. 학칙 RAG 와 실제로 연결한 판은 rag_router_graph.py 입니다.

실행:
    python branch_graph.py            # 설명 + 예시 프롬프트
    python branch_graph.py --plain    # 한 줄 프롬프트 — 경로가 어떻게 바뀌는지 비교
"""

import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

MODEL = os.getenv("MODEL", "gemma3:4b")
llm = ChatOllama(model=MODEL, temperature=0)

ROUTES = ("math", "rag", "chat")
PLAIN = "--plain" in sys.argv


class State(TypedDict):
    question: str
    category: str
    answer: str


# ── 분류 노드 — 5주차 구조화 출력을 쓴다 ★ ───────────────────
class Category(BaseModel):
    """질문 유형."""

    category: Literal["math", "rag", "chat"] = Field(
        description="계산이 필요하면 math, 우리 학교 규정 등 문서 검색이 필요하면 rag, 그 외 잡담은 chat")


PLAIN_PROMPT = ChatPromptTemplate.from_template("질문을 분류하라: {q}")
GUIDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "질문을 세 유형 중 하나로 분류한다.\n"
     "math: 숫자 계산이 필요한 질문 (예: '357 곱하기 4891은?')\n"
     "rag: 우리 학교 학칙·규정처럼 문서를 찾아봐야 답할 수 있는 질문 (예: '휴학은 몇 년까지 돼?')\n"
     "chat: 인사·감정·잡담 (예: '오늘 피곤하다')"),
    ("human", "{q}"),
])


def classify(state: State) -> dict:
    prompt = PLAIN_PROMPT if PLAIN else GUIDED_PROMPT
    try:
        r = (prompt | llm.with_structured_output(Category)).invoke({"q": state["question"]})
        category = r.category
    except Exception as e:  # 소형 모델은 구조화 출력에서도 흔들립니다
        print(f"  [분류] ⚠️ 실패 ({type(e).__name__}) → 기본 경로로 보냅니다")
        category = "chat"
    print(f"  [분류] {category}")
    return {"category": category}


# ── 경로별 노드 ─────────────────────────────────────────────
def calc(state: State) -> dict:
    p = ChatPromptTemplate.from_template("계산 문제다. 단계적으로 풀어라: {q}")
    return {"answer": (p | llm | StrOutputParser()).invoke({"q": state["question"]})}


def search(state: State) -> dict:
    # 여기에 10주차 인덱스 retriever 를 붙이면 RAG 경로가 됩니다 → rag_router_graph.py
    return {"answer": "[검색 경로] 학칙을 찾아 답합니다 — rag_router_graph.py 에서 실제로 연결"}


def chat(state: State) -> dict:
    p = ChatPromptTemplate.from_template("친근하게 한두 문장으로 답하라: {q}")
    return {"answer": (p | llm | StrOutputParser()).invoke({"q": state["question"]})}


# ── 라우팅 ★ ────────────────────────────────────────────────
def route(state: State) -> str:
    """다음에 갈 노드의 '이름' 을 문자열로 돌려준다. ★

    ⚠️ 분류가 틀리면 경로 전체가 틀립니다.
       분류 실패에 대비해 기본 경로(fallback)를 둡니다.
    """
    return state["category"] if state["category"] in ROUTES else "chat"


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("calc", calc)
builder.add_node("search", search)
builder.add_node("chat", chat)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",  # 이 노드 다음에
    route,  # 이 함수로 판단해서
    {"math": "calc", "rag": "search", "chat": "chat"},  # 이 노드로 간다
)
for n in ("calc", "search", "chat"):
    builder.add_edge(n, END)

graph = builder.compile()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"모델: {MODEL} / 분류 프롬프트: {'한 줄 (--plain)' if PLAIN else '설명 + 예시'}\n")

    # ── 서로 다른 경로를 타는 입력 3개 ★ ────────────────────
    for q in ["357 곱하기 4891은?",
              "우리 학교 휴학 규정 알려줘",
              "오늘 기분이 좀 별로야"]:
        print(f"\nQ: {q}")
        out = graph.invoke({"question": q})
        print(f"   경로: {out['category']}")
        print("A:", out["answer"].strip()[:100].replace("\n", " "))

    print("""

관찰 ★

  입력에 따라 다른 노드가 실행됨       분기가 실제로 동작
  실행되지 않은 경로는 비용 0          11주차의 "인사에도 검색이 도는" 낭비를 줄이는 실마리
  라우팅 함수가 문자열을 반환          노드가 아니라 '판단'
  --plain 과 기본의 경로 비교          분류기도 프롬프트 설계가 필요하다 ★

  ⚠️ 357 × 4891 = 1,746,087 — calc 경로의 답이 맞았나요?
     경로를 맞게 타는 것과 답이 맞는 것은 별개입니다 (계산은 도구에 맡기는 편이 안전 → 13주차)
""")


if __name__ == "__main__":
    main()
