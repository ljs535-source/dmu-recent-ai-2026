"""[3교시 / 실습 3] ★★ Reflection 루프 — 생성 → 평가 → 재작성

               ┌──────────────────────────┐
               ▼                          │
   START ─▶ [generate] ─▶ [evaluate] ────┤ "revise"
                              │
                              └─────────▶ END   "end"

  ★ LCEL 로는 아예 못 쓰던 것이 여기서는 딕셔너리 한 줄입니다.
    파이프(|)는 방향이 하나뿐이라 **되돌아가는 화살표를 쓸 문법이 없습니다.**

  과제: 학칙 제29조(제적)를 신입생에게 알리는 **슬랙 공지 한 문단**을 쓴다.
        요구사항(REQUIREMENTS)은 일부러 까다롭게 걸었습니다 — 1회차가 통과하면 루프가 의미 없기 때문입니다.

  ★ 평가 방식 두 가지 (--judge 로 고른다)
     rule  (기본)  규칙으로 검사 — 문장 수 · 글자 수 · 필수 단어 · 마크다운 기호를 **코드가 센다**
     llm           같은 모델이 스스로 채점 (예/아니오) — 원래의 'Self-Critique'
     → 둘을 바꿔 돌려 보세요. **같은 모델이 쓰고 같은 모델이 채점하면** 어떤 일이 생기는지 보입니다.

  ⚠️⚠️ 종료 조건은 두 겹으로 겁니다.
     ① 논리적 종료   상태에 attempts 를 두고 N회 넘으면 END    우리가 설계 ★
     ② 안전망        recursion_limit                        프레임워크의 강제 차단
     정상 종료는 ①로 설계합니다. ②는 버그가 있을 때 에러를 내며 멈추는 장치입니다 (recursion_demo.py).

  ★★ 5주차 Self-Consistency 와 대비
     |          | Self-Consistency (5주) | **Reflection (오늘)**    |
     | 구조     | **병렬**               | **순환** ★               |
     | 방식     | 여러 개를 뽑아 **투표**  | 하나를 **반복해서 다듬음** |
     | 구현     | batch()                | **그래프 엣지**           |
     | 비용     | N배 (한 번에)           | 반복 횟수만큼 (점점)      |

실행:
    python reflection.py                 # 규칙 검사로 평가 (기본)
    python reflection.py --judge llm     # 같은 모델이 스스로 평가
"""

import os
import re
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
MAX_ATTEMPTS = int(os.getenv("WEEK12_MAX_ATTEMPTS", "3"))  # ① 논리적 종료 ★
RECURSION_LIMIT = int(os.getenv("WEEK12_RECURSION_LIMIT", "10"))  # ② 안전망 ★
JUDGE = "llm" if "llm" in sys.argv else "rule"

llm = ChatOllama(model=MODEL, temperature=0.3)

SOURCE = """제29조(제적) 다음 각 호에 해당할 때는 총장이 이를 제적할 수 있다.
1. 휴학기간 종료 후 1개월이 넘도록 이유 없이 복학하지 않은 자
2. 정당한 사유 없이 연속하여 4주를 초과하여 결석한 자
7. 정당한 이유 없이 매학기 소정기간에 등록을 완료하지 못한 자"""

MAX_CHARS = 160
MUST_HAVE = ("제29조", "4주")
REQUIREMENTS = (f"인사말 없이 2~3문장, 공백 포함 {MAX_CHARS}자 이내, 한 문단(줄바꿈 없음), "
                f"'{MUST_HAVE[0]}'와 '{MUST_HAVE[1]}'라는 말을 반드시 포함, "
                "굵은 글씨(**)·머리표(-, *)·제목(#)·이모지 같은 꾸밈 금지")

# 호출 수를 세어 '비용' 을 숫자로 봅니다 ★
CALLS = {"n": 0}


class State(TypedDict):
    draft: str
    critique: str
    passed: bool
    attempts: int


# ── ① 생성 노드 — 지적 사항이 있으면 반영한다 ★ ───────────────
def generate(state: State) -> dict:
    if state.get("critique"):
        p = ChatPromptTemplate.from_template(
            "학칙 조문:\n{src}\n\n이전 초안:\n{draft}\n\n지적 사항:\n{critique}\n\n"
            "지적 사항을 모두 고쳐 슬랙 공지를 다시 써라. 공지 글만 출력하라.\n요구사항: {req}")
        text = (p | llm | StrOutputParser()).invoke({
            "src": SOURCE, "draft": state["draft"],
            "critique": state["critique"], "req": REQUIREMENTS})
    else:
        p = ChatPromptTemplate.from_template(
            "학칙 조문:\n{src}\n\n이 조문을 신입생에게 알리는 슬랙 공지를 써라.\n요구사항: {req}")
        text = (p | llm | StrOutputParser()).invoke({"src": SOURCE, "req": REQUIREMENTS})

    CALLS["n"] += 1
    n = state.get("attempts", 0) + 1
    print(f"\n  [생성 {n}회차] ({len(text.strip())}자) {text.strip()[:70].replace(chr(10), ' ')}...")
    return {"draft": text.strip(), "attempts": n}


# ── ② 평가 노드 — 규칙 검사 / 같은 모델의 자기 평가 ★ ─────────
def sentences(text: str) -> list[str]:
    # \s+	공백(띄어쓰기·줄바꿈)이 한 칸 이상 있는 자리에서 자릅니다.
    # (?<=[.!?。])	단, 바로 앞 글자가 . ! ? 。일 때만 자릅니다. 문장부호는 잘려 나가지 않고 앞 문장에 남습니다
    return [s for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s.strip()]


EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def rule_check(draft: str) -> list[str]:
    """코드가 직접 센다 — 모델의 기분과 상관없이 항상 같은 판정 ★"""
    problems = []
    n = len(sentences(draft))
    if not 2 <= n <= 3:
        problems.append(f"문장이 {n}개다 → 2~3문장으로")
    if len(draft) > MAX_CHARS:
        problems.append(f"{len(draft)}자다 → {MAX_CHARS}자 이내로 줄여라")
    if "\n" in draft:
        problems.append("줄바꿈이 있다 → 한 문단으로")
    for w in MUST_HAVE:
        if w not in draft:
            problems.append(f"'{w}' 가 없다 → 넣어라")
    if "**" in draft or "#" in draft or re.search(r"^\s*[-*]\s", draft, re.M) or EMOJI.search(draft):
        problems.append("꾸밈(**, #, 머리표, 이모지)이 있다 → 빼라")
    return problems


class Critique(BaseModel):
    """초안 평가 결과."""

    passed: bool = Field(description="요구사항을 모두 충족하면 true")
    critique: str = Field(description="부족한 점 한두 문장. 충족하면 'OK'")


def evaluate(state: State) -> dict:
    if JUDGE == "rule":
        problems = rule_check(state["draft"])
        passed, critique = not problems, "; ".join(problems)
    else:
        # ★ 이분 판정(예/아니오) — 7주차 LLM 판정자와 같은 이유로 5점 척도보다 안정적
        p = ChatPromptTemplate.from_template(
            "다음 슬랙 공지가 요구사항을 모두 충족하는지 평가하라.\n요구사항: {req}\n\n공지:\n{draft}")
        try:
            r = (p | llm.with_structured_output(Critique)).invoke(
                {"draft": state["draft"], "req": REQUIREMENTS})
            passed, critique = r.passed, ("" if r.passed else r.critique)
        except Exception as e:  # 구조화 출력이 깨져도 루프가 멈추지 않게
            print(f"  [평가] ⚠️ 실패 ({type(e).__name__}) → 통과로 처리하고 종료합니다")
            passed, critique = True, ""
        CALLS["n"] += 1

    print(f"  [평가 {state['attempts']}회차 · {JUDGE}] {'통과' if passed else '보완'}"
          f"{' — ' + critique[:80] if critique else ''}")
    print(f"  (LLM 호출 누계: {CALLS['n']}회)")
    return {"passed": passed, "critique": critique}


# ── ③ 종료 판단 ★★ ─────────────────────────────────────────
def should_continue(state: State) -> Literal["revise", "end"]:
    if state["passed"]:
        return "end"  # 품질 충족
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"  ⚠️ {MAX_ATTEMPTS}회 도달 — 종료")
        return "end"  # ★ 횟수 상한 (① 논리적 종료)
    return "revise"


# ── ④ 조립 — 여기서 순환이 생긴다 ★ ──────────────────────────
builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("evaluate", evaluate)

builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate", should_continue,
    {"revise": "generate", "end": END},  # ★ "generate" 로 되돌아간다 = 순환
)

graph = builder.compile()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"모델: {MODEL} / 평가: {JUDGE} / 반복 상한 {MAX_ATTEMPTS} / recursion_limit {RECURSION_LIMIT}")
    print(f"요구사항: {REQUIREMENTS}")

    out = graph.invoke(
        {"attempts": 0, "critique": ""},
        config={"recursion_limit": RECURSION_LIMIT},  # ★ ② 안전망
    )

    print("\n" + "=" * 60)
    print(f"총 {out['attempts']}회 시도 / 통과 여부: {out['passed']} / LLM 호출 {CALLS['n']}회")
    print("-" * 60)
    print(out["draft"])
    print("-" * 60)
    print("규칙 검사로 다시 보면:", "; ".join(rule_check(out["draft"])) or "모두 충족")
    print("=" * 60)

    print("""
직접 채울 표 ★

  | 회차 | 글자 수 | 평가(지적 사항) | 호출 수 누계 |
  |  1   |        |                |             |
  |  2   |        |                |             |
  |  3   |        |                |             |

읽어낼 것 ★★

  2회차가 1회차보다 나아졌나?      나아졌다면 Reflection 이 작동한 것
  3회차까지 갔다면 얼마나 나아졌나  반복을 늘릴수록 계속 좋아지는가?
  호출 누계는 어떻게 늘었나        비용은 반복 횟수만큼 늘어난다
  --judge llm 은 몇 회차에 끝났나  같은 모델이 쓰고 같은 모델이 채점 — 마지막 줄 '규칙 검사로 다시 보면' 과 비교 ★★
""")


if __name__ == "__main__":
    main()
