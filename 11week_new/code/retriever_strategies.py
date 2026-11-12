"""[2교시 · 3교시 / 실습 2] ★★ 검색 전략 3종 — MMR · MultiQuery · 하이브리드

  ★ 조립은 되어 있습니다. **파라미터를 바꿔 가며 실험**하세요.
    (특히 하이브리드는 BM25 인덱스를 따로 만들고 묶어야 해서 처음부터 짜기 무겁습니다)

  기본 유사도 검색의 두 한계

     ① 중복       비슷한 것만 잔뜩 온다        → MMR
                  "휴학·복학 규정 전반" 에 비슷한 조각만 몰려 온다

     ② 표현 의존  말을 바꾸면 못 찾는다        → MultiQuery
                  "몇 주를 결석하면 제적" ✅ / "수업을 계속 빠지면 잘리나요" ?
                  사용자는 규정 용어를 모른 채 질문합니다. **그게 정상입니다.**

     (+) 벡터의 약점  조문 번호·숫자          → 하이브리드(BM25)
                  Chroma·FAISS 는 Dense(의미) 전용입니다. 키워드 검색은 BM25 를 따로 만듭니다.

  📌 결론: **"좋은 전략 하나" 가 아니라 "내 질문 유형에 맞는 것"** 입니다.
     그리고 어느 것이 나은지는 **내 데이터로 재봐야 압니다** → 7주차 평가 → 과제 4 ★

실행:
    python retriever_strategies.py                 # 4종(기본·MMR·MQ·하이브리드) 비교표 ★★ ⏱
    python retriever_strategies.py --mmr           # MMR 만 — lambda_mult 를 바꿔 가며 ★
    python retriever_strategies.py --multiquery    # MultiQuery 만 — 다시 쓴 질문을 본다 ★★
    python retriever_strategies.py --hybrid        # 하이브리드만 — weights 를 바꿔 가며 ★
"""

import logging
import re
import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.documents import Document

from questions import QUESTIONS
from rag_common import clip, get_llm, load_chunks, load_store, pad

K = 3


# ── ① 기본 ─────────────────────────────────────────────────
def make_basic(store):
    return store.as_retriever(search_kwargs={"k": K})


# ── ② MMR — 유사도와 다양성의 균형 ★ ─────────────────────────
def make_mmr(store, lambda_mult: float = 0.5, fetch_k: int = 10):
    """기본 : 유사도 상위 k개를 그냥 뽑는다
       MMR  : "질문과 가까우면서 + 이미 뽑은 것과는 다른" 것을 하나씩 뽑는다 ★ (뷔페 접시)

       fetch_k       1차로 넓게 가져올 후보 수 (기본 20 — 학칙은 청크가 14개라 10 으로)
       lambda_mult   1 에 가까울수록 유사도 / 0 에 가까울수록 다양성  ★ 저울
    """
    return store.as_retriever(
        search_type="mmr",  # ★ 한 줄
        search_kwargs={"k": K, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


# ── ③ MultiQuery — 질문을 여러 형태로 다시 쓴다 ★ ─────────────
def make_multiquery(store, llm=None, verbose: bool = False):
    """원 질문을 LLM 이 다시 쓴다 → 각각 검색 → 중복 제거 후 합집합(Unique Union)

       "수업을 계속 빠지면 학교에서 잘리나요?"          (gemma3:4b 가 실제로 쓴 예)
            ① "장기간의 출석 불이행 시 학업 경력 상실에 관한 규정"
            ② "수업 결석이 반복될 경우 학점 인정 기준 미달로 인한 불이익 발생 여부"
            ③ "학적 조치(휴학, 복학, 제적 등)의 사유 및 절차에 따른 학업 중단 시 불이익 규정"

       ⚠️ 합집합이라 k=3 이어도 **3개보다 많이** 돌아올 수 있습니다.
       ⚖️ 대가: **LLM 호출이 추가**됩니다. 지연·비용이 늘어납니다.
    """
    # LangChain 1.x 에서 고급 검색기는 langchain_classic 으로 옮겨졌습니다 ★
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    if verbose:
        # ★★ 다시 쓴 질문이 INFO 로그로 찍힙니다.
        logging.basicConfig()
        logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

    # ⚠️ 기본 프롬프트는 영어라서 질문을 **영어로** 다시 씁니다. 한국어 문서에는 한국어로 쓰게 합니다.
    from langchain_core.prompts import PromptTemplate

    query_prompt = PromptTemplate.from_template(
        "당신은 검색 도우미입니다. 반드시 한국어로 작성하세요.\n"
        "사용자의 질문을 보고, 학칙 문서에서 관련 조문을 찾기 위한 검색 질문 3개를 만드세요.\n"
        "규정집에 나올 법한 표현(휴학·복학·제적·결석·학점 등)으로 바꿔 쓰세요.\n"
        "각 질문은 줄바꿈으로 구분하고, 번호나 다른 말은 붙이지 마세요.\n"
        "원래 질문: {question}"
    )

    return MultiQueryRetriever.from_llm(
        retriever=store.as_retriever(search_kwargs={"k": K}),
        llm=llm or get_llm(),  # ★ 질문을 다시 쓰는 LLM
        prompt=query_prompt,
        include_original=True,  # 원래 질문으로도 검색한다
    )


# ── ④ 하이브리드 — 키워드(Sparse) + 벡터(Dense) ★ ─────────────
def word_tokens(text: str) -> list[str]:
    """단어 단위 — 괄호·문장부호에서 끊는다. '제38조에서' 는 한 덩어리로 남습니다 ⚠️"""
    return re.findall(r"[가-힣A-Za-z0-9]+", text)


def ko_tokens(text: str) -> list[str]:
    """BM25 용 한국어 토큰 — 단어를 **두 글자씩** 겹쳐 자른다 (2-gram).

    ⚠️ 한국어는 조사가 붙습니다. 질문의 '제38조에서' 와 문서의 '제38조' 는
       단어 단위로는 서로 다른 단어라 키워드 검색이 놓칩니다.
       두 글자씩 자르면 '제3' '38' '8조' 가 양쪽에 다 생겨서 맞습니다.
       (실무에서는 형태소 분석기 — 예: Kiwi — 로 조사를 떼어 냅니다)
    """
    out: list[str] = []
    for w in word_tokens(text):
        out += [w[i:i + 2] for i in range(len(w) - 1)] or [w]
    return out


def bm25_docs(chunks) -> list[Document]:
    """⚠️ 구조 분할(10주차 방식 C)은 머리말 '## 제29조(제적)' 을 본문에서 떼어 metadata 로 옮깁니다.
    그대로 두면 BM25 가 '제29조' 라는 글자를 본 적이 없습니다 → 본문 앞에 붙여 줍니다."""
    return [
        Document(page_content=f"{c.metadata.get('article', '')}\n{c.page_content}",
                 metadata=c.metadata)
        for c in chunks
    ]


def make_hybrid(store, chunks, weights=(0.5, 0.5)):
    """BM25(키워드) + 벡터(의미) 의 순위를 가중 합산한다 (Reciprocal Rank Fusion).

       BM25   '제29조' 같은 정확한 토큰에 강함. 의미는 모름
       벡터   '잘리나요' 같은 의역에 강함. 조문 번호는 약함
    """
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    # ── ① BM25 인덱스는 따로 만든다 (문서 텍스트가 필요) ★ ──
    bm25 = BM25Retriever.from_documents(bm25_docs(chunks), preprocess_func=ko_tokens)
    bm25.k = K

    # ── ② 벡터 검색기 ──────────────────────────────────────
    vec = store.as_retriever(search_kwargs={"k": K})

    # ── ③ 묶는다 — 가중치가 핵심 ★ ──────────────────────────
    #    id_key — 같은 청크인지 chunk_id 로 판단한다. BM25 쪽 본문에는 조 이름을 붙였으므로
    #             글자로 비교하면 같은 청크가 두 번 들어옵니다 ⚠️
    return EnsembleRetriever(retrievers=[bm25, vec], weights=list(weights), id_key="chunk_id")


# ── 공통 표시 ───────────────────────────────────────────────
def articles_of(docs) -> list[str]:
    """어느 조가 검색됐는가 — 비교의 단위를 '조' 로 잡으면 표가 읽힙니다 ★"""
    return [d.metadata.get("article", "?") or "?" for d in docs]


def hit(docs, expected: str, top: int = K) -> str:
    """상위 top 건에 정답 근거 조가 포함되었는가 (○/×)

    expected 예: "제29조" / "제25~26조" / "— (문서에 없음)"
    ⚠️ MultiQuery · 하이브리드는 k 보다 많이 돌려줄 수 있어서, 공정하게 앞 top 건만 봅니다.
    """
    if expected.startswith("—"):
        return "—"  # 문서에 없는 내용 — 애초에 맞힐 대상이 아닙니다

    nums = [int(n) for n in re.findall(r"\d+", expected)]
    if len(nums) == 2 and "~" in expected:  # 범위 표기
        nums = list(range(nums[0], nums[1] + 1))

    got = articles_of(docs[:top])
    return "○" if any(a.startswith(f"제{n}조") for a in got for n in nums) else "×"


def compare_all() -> None:
    store = load_store()
    chunks = load_chunks(store)

    strategies = {
        "기본": make_basic(store),
        "MMR": make_mmr(store),
        "MultiQuery": make_multiquery(store),
        "하이브리드": make_hybrid(store, chunks),
    }

    print(f"\n질문 {len(QUESTIONS)}개 × 전략 {len(strategies)}종 — 상위 {K}건에 정답 조가 있는가")
    print("⏱ MultiQuery 는 질문마다 LLM 호출이 추가되어 느립니다.\n")

    print("  " + pad("질문", 46) + pad("유형", 14)
          + "".join(pad(n, 12, ">") for n in strategies))
    print("  " + "─" * 108)

    for q, kind, expected in QUESTIONS:
        cells = []
        for retr in strategies.values():
            try:
                cells.append(hit(retr.invoke(q), expected))
            except Exception as e:
                cells.append(f"!{type(e).__name__[:6]}")
        print("  " + pad(clip(q, 44), 46) + pad(kind, 14)
              + "".join(pad(c, 12, ">") for c in cells))

    print("""
  ○ = 상위 3건에 정답 근거 조가 포함됨 / × = 없음 / — = 문서에 없는 내용

  ★ 표를 보고 적어 보세요
     · 어느 유형에서 기본이 이미 충분했는가?
     · 기본이 ×인데 다른 전략이 ○로 바꾼 칸은 어디인가? 그 전략은 무엇을 풀었는가?
     · 거꾸로, 기본은 ○인데 전략을 쓰자 ×가 된 칸은? (공짜가 아닙니다)

  ⚠️ 같은 코드라도 모델(gemma3:4b · 1b)과 실행마다 MultiQuery 칸이 바뀔 수 있습니다.
""")


def demo_mmr() -> None:
    store = load_store()
    q = "휴학과 복학 관련 규정을 전반적으로 알려주세요"  # 넓은 질문 ★
    print(f"질문: {q}\n")

    print("  [기본]       ", articles_of(make_basic(store).invoke(q)))
    for lm in (0.9, 0.7, 0.5, 0.2):
        docs = make_mmr(store, lambda_mult=lm).invoke(q)
        print(f"  [MMR λ={lm}]  ", articles_of(docs))

    print("""
  ★ λ 를 낮출수록 '서로 다른 것' 이 섞여 들어옵니다.
  ⚖️ **공짜가 아닙니다** — 다양성을 얻으려면 **질문과의 관련도를 일부 내줍니다.**
     문서가 작으면(학칙 14청크) 금방 질문과 상관없는 조까지 끌려옵니다.
     → λ 는 문서 크기와 질문 유형에 맞춰 골라야 합니다.
""")


def demo_multiquery() -> None:
    store = load_store()
    retr = make_multiquery(store, verbose=True)  # ★ INFO 로그로 다시 쓴 질문이 찍힙니다

    q = "수업을 계속 빠지면 학교에서 잘리나요?"  # 구어체 ★
    print(f"질문: {q}")
    print("(아래 INFO 로그가 '모델이 다시 쓴 질문' 입니다 ★★)\n")

    docs = retr.invoke(q)
    print("\n  [MultiQuery]   ", articles_of(docs), f"← {len(docs)}개 (합집합)")
    print("  [기본 검색]    ", articles_of(make_basic(store).invoke(q)))

    print("""
  ★★ "모델이 내 질문을 이렇게 바꿔서 검색했구나" 를 보는 순간 이 전략이 이해됩니다.
     LangSmith 추적에서도 **LLM Run 이 하나 더 늘어난 것**이 보입니다.

  ⚖️ 대가: LLM 호출이 추가됩니다. 지연·비용이 늘어납니다.

  💡 13주차 예고: MultiQuery 는 **항상** 여러 질문을 만듭니다.
     Agentic RAG 는 **실패했을 때만** 다시 씁니다. 같은 문제, 다른 해법입니다. ★
""")


def demo_hybrid() -> None:
    store = load_store()
    chunks = load_chunks(store)

    cases = [
        ("제38조에서 정한 출석 기준은?", "고유명사 ★"),
        ("수업을 계속 빠지면 학교에서 잘리나요?", "구어체 ★"),
    ]

    from langchain_community.retrievers import BM25Retriever

    for q, kind in cases:
        print(f"\n질문 [{kind}]: {q}")
        print(f"  단어 토큰 : {word_tokens(q)}")
        print(f"  2-gram    : {ko_tokens(q)}")
        for name, fn in [("단어", word_tokens), ("2-gram", ko_tokens)]:
            b = BM25Retriever.from_documents(bm25_docs(chunks), preprocess_func=fn)
            b.k = K
            print(f"  [BM25만 · {name:6s}]       ", articles_of(b.invoke(q)))
        print("  [벡터만]                ", articles_of(make_basic(store).invoke(q)))
        for w in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
            docs = make_hybrid(store, chunks, weights=w).invoke(q)
            print(f"  [BM25 {w[0]} / 벡터 {w[1]}]    ", articles_of(docs)[:K])

    print("""
  ★ 단어 토큰이면 BM25 가 조사 때문에 '제38조에서' 를 못 맞힙니다. 2-gram 이면 맞힙니다.
  ★ 고유명사 질문과 구어체 질문에서 **잘 듣는 가중치가 서로 다릅니다.**
    → "하나의 정답 가중치는 없다" 가 이 실습의 결론입니다.

  💡 rank_bm25 는 순수 파이썬 CPU 라이브러리라 VRAM 을 쓰지 않습니다.
""")


def main() -> None:
    if "--mmr" in sys.argv:
        demo_mmr()
    elif "--multiquery" in sys.argv:
        demo_multiquery()
    elif "--hybrid" in sys.argv:
        demo_hybrid()
    else:
        compare_all()


if __name__ == "__main__":
    main()
