"""[3교시 / 실습 3] ★ Lost in the Middle — 정답 문서의 '위치' 를 바꿔 가며 본다

  핵심 질문: 관련 문서를 10개 넣으면 3개보다 잘 답합니까?

     긴 컨텍스트에서 모델이 정보를 반영하는 정도 (Liu et al., 2023)

     위치:   1    2    3    4    5    6    7    8    9   10
     반영:  ███  ██▓  ██   █▓   █    █    █▓   ██   ██▓  ███
            높음                 낮음 ⚠️                   높음
            └── 앞 ──┘      └─ 중간 ─┘        └── 뒤 ──┘

  ★ 실험 요령: **검색을 쓰지 않고 문서 순서를 직접 정합니다.**
    검색에 맡기면 순서가 매번 달라져 실험이 성립하지 않습니다.

  ⚠️ 방해 문서는 전부 **실제 학칙의 다른 조문**입니다.
     "3주 이내 복학" · "4분의 1 이상" 처럼 숫자가 비슷한 문장이 섞여 있어 모델이 실제로 헷갈립니다.

  ⚠️ 진단 순서가 하나 늘었습니다
     ① 검색 결과에 답이 있는가?     → 없으면 **검색 문제**
     ② 있는데 틀렸다면 → **몇 번째에 있었는가?** ★ ← 오늘 추가
     ③ 그래도 아니면 → 프롬프트·모델 문제

  📌 "검색이 정답을 가져왔다" 와 "모델이 그것을 봤다" 는 다른 이야기입니다.

  ★ 모델마다 다릅니다 — 같은 실험을 두 모델로 돌려 보세요.
     이 현상이 얼마나 강한지는 모델·문맥 길이·문서 내용에 따라 다릅니다. 내 모델로 재 봐야 압니다.

실행:
    python lost_in_middle.py                       # 위치 실험 + 개수 실험 (문서 10개)
    python lost_in_middle.py --repeat 3 --n 20     # 3번 반복 · 문서 20개 ★
    python lost_in_middle.py --repeat 3 --n 20 --model llama3.2:1b   # 작은 모델과 비교 ★
        (ollama pull llama3.2:1b 가 필요합니다 — 약 1.3GB)
"""

import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from questions import LIM_ANSWER_DOC, LIM_ANSWER_KEY, LIM_QUESTION
from rag_common import get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "아래 <context> 안의 문서들만 근거로 한 문장으로 답하라. 없으면 '모름'이라고 답하라."),
    ("human", "<context>\n{context}\n</context>\n\n질문: {question}"),
])

# 관련 없는(그러나 **그럴듯한**) 방해 문서 — 실제 학칙 제25~43조에서 옮김 ★
DISTRACTORS = [
    "제26조(복학) ① 학기 개학 후 3주 이내에 복학이 가능한 자는 금번 학기에 복학할 수 있다.",
    "제25조(휴학) ① 질병 등으로 총 수업일수의 4분의 1 이상 수업할 수 없을 때는 총장의 허가를 받아 휴학할 수 있다.",
    "제38조(성적평가) ⑥ 각 교과목 총수업 시간 수의 4분의 3 이상을 출석한 자만 학업성적을 인정할 수 있다.",
    "제25조(휴학) ③ 휴학기간은 부득이한 경우를 제외하고는 1년을 초과하지 못한다.",
    "제27조(전과) ① 전과는 1학년 2학기 및 2학년 1학기에 한해서 허가할 수 있다.",
    "제42조(졸업) ① 수업연한이 2년인 학과의 졸업 이수학점은 75학점 이상, 3년인 학과는 110학점 이상으로 한다.",
    "제28조(자퇴) 자퇴하고자 하는 자는 보증인 연서로 사유를 명시하여 총장의 허가를 받아야 한다.",
    "제40조(성적의 취소) ① 이미 인정된 성적이 과오 또는 부정행위에 의한 것일 때에는 이를 취소한다.",
    "제42조의3(수료) ① 전문학사과정의 1학년 수료에 필요한 학점은 38학점 이상이다.",
    "제26조(복학) ② 일반 휴학 중 군에 입대하여 제적된 자가 제대 후 복학하는 경우 군입대 휴학으로 인정할 수 있다.",
    "제41조(학사경고) 성적이 불량하고 탈락이 예상되는 자는 경고하여야 하며 그 기준은 총장이 따로 정한다.",
    "제37조(평가시험) ① 성적평가는 주관식 시험과 객관식 시험을 병용한다.",
    "제25조(휴학) ③ 창업휴학을 하고자 휴학연기원서를 제출하는 경우에는 2년까지 허용한다.",
    "제38조(성적평가) ③ 각 교과목의 강좌별 급제는 D0급 이상으로 한다.",
    "제42조(졸업) ② 학사학위전공심화과정의 졸업 이수학점은 2년인 학과의 경우 55학점 이상으로 한다.",
    "제25조(휴학) ④ 휴학한 자는 휴학기간 중 학적을 보유하며 휴학기간 만료와 동시에 복학한다.",
    "제27조(전과) ④ 전과한 자의 기 취득학점은 전입학과의 학점으로 인정할 수 있다.",
    "제38조(성적평가) ④ 급낙평가는 P를 급제, NP를 낙제로 하고 평점평균 산출에서는 제외한다.",
    "제43조(학점은행제) ① 총장은 「학점인정 등에 관한 법률」에 따라 학습과정을 설치하여 운영할 수 있다.",
]


def build(pos: int, n_docs: int = 10) -> str:
    """정답 문서를 pos 번째(1부터)에 끼워 넣는다."""
    docs = DISTRACTORS[: n_docs - 1].copy()
    docs.insert(pos - 1, LIM_ANSWER_DOC)
    return "\n\n".join(f"[문서 {i}] {d}" for i, d in enumerate(docs, 1))


def judge(answer: str) -> bool:
    """정답 판정 — '4주' 가 들어 있는가 (단순하게 갑니다)"""
    return LIM_ANSWER_KEY in answer


def run(chain, pos: int, n: int, repeat: int) -> tuple[str, str]:
    oks, last = 0, ""
    for _ in range(repeat):
        last = chain.invoke({"context": build(pos, n), "question": LIM_QUESTION})
        oks += judge(last)
    rate = f"{oks}/{repeat}" if repeat > 1 else ("✅" if oks else "❌")
    return rate, last.strip().replace("\n", " ")[:50]


def main() -> None:
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10
    n = max(3, min(n, len(DISTRACTORS) + 1))

    if "--model" in sys.argv:
        from langchain_ollama import ChatOllama

        model = sys.argv[sys.argv.index("--model") + 1]
        llm = ChatOllama(model=model, temperature=0)
    else:
        llm = get_llm()
        model = llm.model
    chain = PROMPT | llm | StrOutputParser()

    print(f"모델: {model}")
    print(f"질문: {LIM_QUESTION}")
    print(f"정답: {LIM_ANSWER_DOC}\n")

    print(f"── 실험 ① 정답의 '위치' 를 바꾼다 (문서 {n}개 고정) ★ ──")
    mid = (n + 1) // 2
    for pos in (1, mid, n):
        rate, ans = run(chain, pos, n, repeat)
        tag = " ← 중간 ★" if pos == mid else ""
        print(f"  정답 위치 {pos:2d}번  →  {rate}  {ans}{tag}")

    print("\n── 실험 ② 문서 '개수' 를 줄인다 (정답은 항상 가운데) ★ ──")
    for k in sorted({n, 5, 3}, reverse=True):
        rate, ans = run(chain, (k + 1) // 2, k, repeat)
        print(f"  k={k:2d}  →  {rate}  {ans}")

    print("""
  ★ 표에 옮겨 적으세요
     | 정답 위치 | 맞았나 |      | 문서 개수 | 맞았나 |
     | 앞        |        |      | k=10     |        |
     | 가운데 ★  |        |      | k=5      |        |
     | 뒤        |        |      | k=3      |        |

  읽어 낼 것
     앞·뒤는 맞고 가운데는 틀림   → Lost in the Middle 확인 ★★
     k 를 줄이니 맞음             → 적게 넣는 것이 답일 때가 있다
     전부 맞음                    → 이 모델·문서 길이에서는 안 나타남 → --model llama3.2:1b 와 비교

  대응 3가지
     개수를 줄인다             k=10 → 3~5               가장 간단하고 효과적 ★
     재정렬해서 양끝에 배치     중요한 것을 1번과 마지막    Re-ranking
     압축한다                  관련 부분만 남긴다          Contextual Compression

  ⚖️ k 를 늘리면 **놓치지 않기(재현율)** 는 오르지만 **반영되기** 는 떨어집니다. 또 트레이드오프입니다.
  ⚠️ LLM 은 실행마다 답이 흔들립니다. 한 번 결과로 결론 내리지 말고 --repeat 3 으로 확인하세요.
""")


if __name__ == "__main__":
    main()
