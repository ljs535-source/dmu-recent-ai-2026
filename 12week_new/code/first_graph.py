"""[1교시 판서 · 3교시 실습 1] 첫 그래프 — 노드 2개를 직렬로 잇는다

              ┌──────────── State (공용 상태) ────────────┐
              │  {"question": ..., "answer": ..., ...}    │
              └───────────▲──────────────▲────────────────┘
                          │ 읽고 쓴다     │
                    ┌─────┴────┐   ┌─────┴────┐
      START ──Edge──▶│  Node A  │──▶│  Node B  │──Edge──▶ END
                    └──────────┘   └──────────┘

  | 부품     | 정체       | 한 줄 정의                          |
  | State    | TypedDict  | 노드들이 **함께 읽고 쓰는 공용 저장소** ★ |
  | Node     | **함수**   | 상태를 받아 **갱신할 부분만** 돌려준다   |
  | Edge     | 연결       | **다음에 어디로 갈지**                |

  ★ 체인과의 결정적 차이는 State 입니다.
    체인은 값이 **통과**하지만, 그래프는 값이 **머물러 있고** 노드들이 그것을 고칩니다.
    그래서 되돌아가도 **그동안의 작업이 남아 있습니다.**

  ⚖️ 직렬 처리만 할 거면 **LCEL 이 더 짧고 읽기 쉽습니다.**
     그래프는 **분기와 순환이 필요할 때** 값을 합니다. 다음 실습부터가 본론입니다. ★

실행:
    python first_graph.py
"""

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()  # ★ 반드시 맨 위

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

MODEL = os.getenv("MODEL", "gemma3:4b")
llm = ChatOllama(model=MODEL, temperature=0)


# ── ① 상태 스키마 ───────────────────────────────────────────
class State(TypedDict):
    question: str
    keywords: str
    answer: str


# ── ② 노드 = 함수 ★ ─────────────────────────────────────────
def extract_keywords(state: State) -> dict:
    p = ChatPromptTemplate.from_template("다음 질문의 핵심 키워드 3개만 쉼표로: {q}")
    kw = (p | llm | StrOutputParser()).invoke({"q": state["question"]})
    print(f"  [노드1] 키워드 = {kw.strip()[:60]}")
    return {"keywords": kw.strip()}  # ★ 갱신할 키만 — question 은 그대로 유지된다


def write_answer(state: State) -> dict:
    p = ChatPromptTemplate.from_template(
        "키워드({kw})를 참고해 질문에 3문장으로 답하라.\n질문: {q}")
    ans = (p | llm | StrOutputParser()).invoke(
        {"kw": state["keywords"], "q": state["question"]})
    print("  [노드2] 답변 작성 완료")
    return {"answer": ans}


# ── ③ 조립 ──────────────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("extract", extract_keywords)  # ① 노드 등록
builder.add_node("write", write_answer)

builder.add_edge(START, "extract")  # ② 엣지 연결
builder.add_edge("extract", "write")
builder.add_edge("write", END)

graph = builder.compile()  # ③ 컴파일 → Runnable 이 된다 ★


def main() -> None:
    print(f"모델: {MODEL}\n")

    # ── ④ 실행 ─────────────────────────────────────────────
    out = graph.invoke({"question": "LangGraph를 왜 쓰나요?"})

    print("\n최종 상태 키:", list(out.keys()))  # ★ 전부 남아 있다
    print("-" * 60)
    print(out["answer"].strip())
    print("-" * 60)

    # ★ compile() 의 결과도 Runnable 입니다 — batch/stream 을 그대로 씁니다
    print("\ngraph 의 타입:", type(graph).__name__)
    print("invoke / batch / stream 을 갖고 있는가:",
          all(hasattr(graph, m) for m in ("invoke", "batch", "stream")))

    print("""
관찰 포인트 ★

  답의 내용도 읽어 보세요 — 모델이 LangGraph 를 정말 알고 답했나요?
        → 모르는 것을 그럴듯하게 지어내는 것은 그래프로 바꿔도 그대로입니다 (10주차 RAG 가 필요했던 이유)

  최종 결과에 question·keywords·answer 가 **모두 있다**
        → "체인이라면 앞의 값은 사라졌습니다" ★
  노드가 **갱신할 키만** 반환
        → 나머지는 LangGraph 가 알아서 합칩니다
          (5주차 RunnablePassthrough.assign 과 비슷한 발상)
  compile() 의 결과도 Runnable
        → 5주차에 배운 규약이 그래프에도 적용됩니다.
          그래서 체인 안에 그래프를 끼울 수도 있습니다 ★
          (14주차 '서브그래프' 가 이 성질을 씁니다)

  ⚠️ 이 정도면 **체인이 더 간단합니다.** 맞습니다.
     **분기·순환이 없으면 LCEL 을 쓰세요.** ★
     다음 실습(조건부 엣지)부터가 그래프를 쓰는 이유입니다.
""")


if __name__ == "__main__":
    main()
