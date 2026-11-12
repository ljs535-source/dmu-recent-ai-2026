"""[2교시 / 3절] 개념 4종 — 어떤 문제를 푸는가

  🎯 여기서는 **"어떤 상황에 무엇을 쓰는가"** 를 잡습니다.
     3종(MMR·MultiQuery·하이브리드)은 retriever_strategies.py 에서 직접 돌리고,
     이 4종은 개념으로 정리합니다. (7종을 모두 실습하면 어느 것도 제대로 끝나지 않습니다)

  이 파일은 **읽는 자료**입니다. --run 을 붙이지 않는 한 아무것도 호출하지 않습니다. ★

실행:
    python concepts_4.py                  # 4종 설명 + 7종 한눈에 표 (호출 없음 · 무료) ★
    python concepts_4.py --run compress   # (선택) Compression — EmbeddingsFilter 로 문서 버리기
    python concepts_4.py --run parent     # (선택) Parent Document — 작게 검색하고 크게 답하기
"""

import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")


# ── 3-1. Contextual Compression ─────────────────────────────
COMPRESSION = """
── ① Contextual Compression — 가져온 것에서 관련 부분만 ──────────

   비유: 책 5쪽을 통째로 건네기  vs  **형광펜 친 문장만 오려서 건네기**

   검색 결과: 제29조 청크 (8개 호)
        │   그중 질문 «몇 주 결석하면 제적?» 과 관련된 것은 2호 한 줄뿐
        ▼
   관련 부분만 남긴다 → 프롬프트가 짧아진다 → 노이즈↓ · Lost in the Middle 완화 ★

   두 가지 방법
     LLMChainExtractor   LLM 에게 "필요한 부분만 뽑아라"   정확 · **문서마다 LLM 호출 → 느림**
     EmbeddingsFilter    질문과 유사도가 기준보다 낮은 문서를 통째로 버림
                                                          빠름 · 문장 단위로는 못 자름

   # 개념 코드 (LangChain 1.x 경로)
   from langchain_classic.retrievers import ContextualCompressionRetriever
   from langchain_classic.retrievers.document_compressors import LLMChainExtractor

   compressor = LLMChainExtractor.from_llm(llm)
   retriever_cc = ContextualCompressionRetriever(
       base_compressor=compressor, base_retriever=vec)

   푸는 문제 : 긴 청크에 관련 없는 내용이 섞임
   대가      : LLM 호출 추가 (Extractor 는 문서 수만큼!)
"""

# ── 3-2. Parent Document ★★ ─────────────────────────────────
PARENT = """
── ② Parent Document — 작게 검색하고 크게 답한다 ★★ ──────────────

   ★ 10주차 청크 크기 딜레마의 정면 해법입니다.

   [10주차의 딜레마]
     작게 자르면 → 검색은 정확한데 문맥이 없다
     크게 자르면 → 문맥은 있는데 검색이 부정확하다
                        │
                        ▼
   [Parent Document]
     자식 청크(작게)로 검색한다        ← 검색 정확도 ✅
     부모 청크(크게)를 돌려준다        ← 문맥 확보 ✅  ★

   부모: 제29조 전체 (8개 호)
     ├ 자식1: "1. 휴학기간 종료 후 1개월이 넘도록..."
     ├ 자식2: "2. 정당한 사유 없이 연속하여 4주를 초과하여 결석한 자"  ← 이걸로 검색
     └ 자식3: "3. 퇴학에 해당하는 징계처분을 받은 자 ..."
                    │  자식2가 검색되면
                    ▼
              부모(제29조 전체)를 프롬프트에 넣는다 ★

   푸는 문제 : **작게 자르면 문맥 소실 / 크게 자르면 노이즈** ★★
   대가      : 저장 구조가 복잡해짐 (부모-자식 두 벌 관리)
"""

# ── 3-3. Self-Query ─────────────────────────────────────────
SELF_QUERY = """
── ③ Self-Query — 자연어에서 필터를 자동 추출 ────────────────────

   사용자: "성적·졸업 장에서 출석 기준 알려줘"
        │
        │  LLM 이 둘로 나눈다  ★
        ▼
   검색어  : "출석 기준"
   필터    : {"chapter": "제7장 성적평가 및 졸업"}   ← 1교시 필터 검색으로 ★
        ▼
   metadata 필터 + 유사도 검색을 함께 수행

   푸는 문제 : 사용자가 **필터 문법을 모른다**
   대가      : metadata 스키마 정의 필요 / **파싱 실패 가능** ⚠️

   ★ 10주차 metadata 설계가 여기서도 회수됩니다.
     chapter 를 안 심었으면 **Self-Query 도 못 씁니다.**
   ⚠️ 파싱 실패 가능성 → 5주차 구조화 출력(with_structured_output) 으로 안정화 ★
"""

# ── 3-4. Re-ranking ─────────────────────────────────────────
RERANK = """
── ④ Re-ranking — 넓게 가져와 다시 줄 세운다 ─────────────────────

   1차 : 벡터 검색으로 20건을 '넓게' 가져온다   (빠르지만 순위는 대충)
        ▼
   2차 : 재정렬 모델이 질문-문서 쌍을 **하나씩 함께 읽고** 채점 → 상위 5건 ★
        ▼
   상위권 정확도가 올라간다

   왜 2차가 더 정확한가
     1차(임베딩)  질문과 문서를 **따로** 벡터로 만들고 거리만 잰다   → 빠름
     2차(재정렬)  질문과 문서를 **한 번에 같이** 읽고 점수를 낸다    → 느리지만 정확

   푸는 문제 : **1차 검색 상위권의 정확도가 낮음**
   대가      : 모델 추가 → **지연 증가 · VRAM 추가** ⚠️
              실습 PC(8GB)는 gemma3:4b + 임베딩 + KV 캐시만으로도 빠듯해서 이번 수업에서는 돌리지 않습니다.
"""

TABLE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7종 한눈에 ★★

  | 전략                    | 다루는 방식 | 푸는 문제                          | 대가              |
  |-------------------------|-----------|-----------------------------------|------------------|
  | **MMR**                 | 실습      | 검색 결과가 **서로 중복됨**          | 관련도 일부 희생    |
  | **MultiQuery**          | 실습      | **질문 표현이 문서와 다름**          | LLM 호출 추가      |
  | **하이브리드**           | 실습      | **조문 번호·숫자**를 벡터가 놓침      | 가중치·토큰 조정    |
  | Contextual Compression  | 개념      | 긴 청크에 관련 없는 내용이 섞임       | LLM 호출 추가      |
  | **Parent Document** ★★  | 개념      | **작게=문맥 소실 / 크게=노이즈**     | 저장 구조 복잡      |
  | Self-Query              | 개념      | 사용자가 **필터를 직접 못 씀**       | 스키마 정의·파싱 실패|
  | Re-ranking              | 개념      | **1차 상위권 정확도가 낮음**         | 모델 추가(지연·VRAM)|

📌 표를 외우기보다 **"이 문제에는 → 이 전략"** 으로 답하는 연습을 하세요.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def run_compress() -> None:
    """EmbeddingsFilter — LLM 없이 임베딩만으로 '관련 없는 문서를 버린다'."""
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

    from rag_common import get_emb, load_store

    store = load_store()
    base = store.as_retriever(search_kwargs={"k": 5})
    q = "몇 주를 결석하면 제적되나요?"
    print(f"질문: {q}\n")
    print("  [기본 k=5]           ", [d.metadata.get("article") for d in base.invoke(q)])

    for th in (0.45, 0.55):
        cc = ContextualCompressionRetriever(
            base_compressor=EmbeddingsFilter(embeddings=get_emb(), similarity_threshold=th),
            base_retriever=base,
        )
        print(f"  [EmbeddingsFilter {th}]", [d.metadata.get("article") for d in cc.invoke(q)])

    print("""
  ★ 기준(similarity_threshold, 코사인 유사도)보다 낮은 문서를 **통째로** 버립니다.
    청크 안의 문장을 잘라 내지는 못합니다 — 그건 LLMChainExtractor 의 일입니다.
""")


def run_parent() -> None:
    """Parent Document — 자식(작게)으로 찾고 부모(크게)를 돌려준다."""
    from langchain_chroma import Chroma
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_common import KO_SEPARATORS, get_emb, load_chunks

    docs = load_chunks()  # 10주차 청크 = 조 하나씩 → 이것을 부모로 씁니다

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80, chunk_overlap=0, separators=KO_SEPARATORS)  # ★ 작게 검색

    retriever = ParentDocumentRetriever(
        vectorstore=Chroma(collection_name="week11_parent", embedding_function=get_emb()),
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
    )
    retriever.add_documents(docs)

    q = "몇 주를 결석하면 제적되나요?"
    print(f"질문: {q}\n")

    print("── 자식 청크로 검색한 결과 (작다 — 정확하다) ─────────")
    for d in retriever.vectorstore.similarity_search(q, k=2):
        print(f"  ({len(d.page_content):3d}자) {d.page_content[:70]!r}")

    print("\n── 실제로 돌려주는 것: 부모 청크 (크다 — 문맥이 있다) ★")
    for d in retriever.invoke(q)[:2]:
        print(f"  ({len(d.page_content):3d}자) {d.metadata.get('article')} "
              f"{d.page_content[:60]!r}...")

    print("""
  ★★ "작게 검색하고 크게 답한다."
     검색은 작은 자식으로 정확하게, 답변은 조 전체(부모)로 문맥을 갖춰서.
""")


def main() -> None:
    if "--run" in sys.argv:
        target = sys.argv[sys.argv.index("--run") + 1]
        {"compress": run_compress, "parent": run_parent}.get(
            target, lambda: print(f"⚠️ '--run {target}' 은 없습니다. (compress · parent)"))()
        return

    print(COMPRESSION)
    print(PARENT)
    print(SELF_QUERY)
    print(RERANK)
    print(TABLE)


if __name__ == "__main__":
    main()
