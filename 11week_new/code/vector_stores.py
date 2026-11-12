"""[1교시 / 실습 1-①] 벡터 저장소 두 개를 나란히 — FAISS 와 Chroma

  ★ 같은 청크 14개를 두 그릇에 담고, 같은 질문을 던져 봅니다.

       FAISS   = 검색 **부품**(라이브러리)   벡터만 갖고, 글·metadata 는 바깥 docstore 에
       Chroma  = 임베딩 **데이터베이스**     글·metadata·벡터를 한 곳(chroma.sqlite3)에

  ★ 인터페이스가 같아서 바꿔 끼울 수 있습니다 — 4주차 모델 교체와 같은 구조.
    similarity_search · similarity_search_with_score · as_retriever 가 둘 다 있습니다.
    그런데 **속 동작은 다릅니다** — 이 파일에서 직접 확인합니다.

       기본 검색      FAISS = IndexFlatL2 (전부 비교 · 정확)   Chroma = HNSW (그래프 · 근사)
       점수           둘 다 L2 거리 — **작을수록 가깝다**
       metadata 필터  FAISS = **꺼낸 뒤 거른다** (fetch_k)     Chroma = **거른 뒤 찾는다** (where)

실행:
    python vector_stores.py            # 두 저장소의 정체 + Chroma 저장 폴더 보기
    python vector_stores.py --scores   # 같은 질문, 두 저장소의 점수 + 거리 척도(l2 · cosine) ★
    python vector_stores.py --filter   # chapter 필터 — 꺼낸 뒤 거르기 vs 거른 뒤 찾기 ★★
"""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from rag_common import clip, get_emb, load_chunks, load_store

CHROMA_DIR = Path(__file__).parent / "chroma_db"  # ⚠️ 실행할 때마다 새로 만듭니다
QUESTION = "몇 주를 결석하면 제적되나요?"


def short(doc) -> str:
    """'제29조(제적)' 처럼 조 이름만 — 표가 읽히게 합니다."""
    return doc.metadata.get("article", "?") or "?"


def build_chroma(chunks, space: str = "l2", name: str = "week11"):
    """Chroma 를 디스크(chroma_db/)에 만든다. space = 거리 척도 (l2 · cosine · ip)

    ⚠️ 컬렉션 이름은 영문·숫자 3~512자여야 합니다 ("t" 같은 한 글자는 거절됩니다).
    """
    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name)  # 다시 실행해도 문서가 두 벌로 쌓이지 않게
    except Exception:
        pass

    store = Chroma(
        collection_name=name,
        embedding_function=get_emb(),
        client=client,
        collection_configuration={"hnsw": {"space": space}},  # ★ 거리 척도
    )
    store.add_documents(chunks)
    return store


def show_tree(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        size = f"{p.stat().st_size:>10,d} B" if p.is_file() else "  (폴더)"
        print(f"    {str(rel):60s}{size}")


def demo_identity() -> None:
    faiss_store = load_store()
    chunks = load_chunks(faiss_store)

    print("\n── ① FAISS — 검색 부품 ─────────────────────────────")
    print(f"  인덱스 종류   : {type(faiss_store.index).__name__}   ← Flat = 전부 비교 (정확)")
    print(f"  벡터 수       : {faiss_store.index.ntotal}")
    print(f"  벡터 차원     : {faiss_store.index.d}")
    print(f"  글·metadata   : {type(faiss_store.docstore).__name__}  ← FAISS 바깥에 따로 ★")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    chroma = build_chroma(chunks)
    conf = chroma._collection.configuration.get("hnsw", {})

    print("\n── ② Chroma — 임베딩 데이터베이스 ─────────────────────")
    print(f"  저장 위치     : {CHROMA_DIR}")
    print(f"  문서 수       : {chroma._collection.count()}")
    print(f"  인덱스·거리   : HNSW · space={conf.get('space')}   ← 기본값 l2")
    print(f"  HNSW 설정     : max_neighbors={conf.get('max_neighbors')} · "
          f"ef_search={conf.get('ef_search')} · sync_threshold={conf.get('sync_threshold')}")
    print("\n  폴더 안 ↓")
    show_tree(CHROMA_DIR)

    print("""
  ★ 폴더를 보세요.
     chroma.sqlite3    글 · metadata · 벡터가 들어 있는 데이터베이스 파일
     <긴 이름 폴더>/   HNSW 그래프 파일이 생길 자리 — 지금은 비어 있습니다
                       벡터가 sync_threshold(1000개)를 넘어야 그래프 파일
                       (data_level0.bin · link_lists.bin …)을 따로 씁니다.
     → 청크 14개짜리 학칙은 sqlite 파일 하나로 충분하다는 뜻입니다.

  ★ 둘 다 같은 질문에 답할 수 있습니다:  python vector_stores.py --scores
""")


def demo_scores() -> None:
    faiss_store = load_store()
    chunks = load_chunks(faiss_store)
    chroma_l2 = build_chroma(chunks, "l2", "week11")
    chroma_cos = build_chroma(chunks, "cosine", "week11_cosine")

    print(f"\n질문: {QUESTION}\n")
    rows = [
        ("FAISS  (IndexFlatL2)", faiss_store),
        ("Chroma (HNSW · l2)", chroma_l2),
        ("Chroma (HNSW · cosine)", chroma_cos),
    ]
    for label, store in rows:
        print(f"  [{label}]")
        for doc, score in store.similarity_search_with_score(QUESTION, k=3):
            print(f"    {score:7.4f}  {short(doc):18s} {clip(doc.page_content, 36)}")
        print()

    print("""  ★ 읽는 법
     · 세 줄 모두 **거리**입니다 — 작을수록 가깝습니다. (유사도처럼 클수록 좋은 게 아닙니다)
     · FAISS 와 Chroma(l2) 의 숫자가 거의 같습니다. 둘 다 '제곱한 L2 거리' 이기 때문입니다.
     · cosine 거리는 0~2 범위 — 0 이면 방향이 같다(같은 말), 1 이면 관계없음, 2 면 반대.
     · cosine 값이 l2 값의 **정확히 절반**입니다. 임베딩 벡터의 길이가 모두 1(정규화)이라서
       '제곱 L2 거리 = 2 × cosine 거리' 가 됩니다. 그래서 척도를 바꿔도 **순위가 같습니다.**

  ★ 1위와 3위의 차이가 크면 검색이 확신을 갖고 있고, 비슷하면 헤매는 중입니다.
""")


def demo_filter() -> None:
    faiss_store = load_store()
    chunks = load_chunks(faiss_store)
    chroma = build_chroma(chunks)

    ch7 = next(c.metadata["chapter"] for c in chunks if c.metadata["chapter"].startswith("제7장"))
    flt = {"chapter": ch7}
    print(f"\n질문: {QUESTION}")
    print(f"필터: {flt}   ← 10주차에 심어 둔 metadata ★\n")

    print("  [필터 없음 — FAISS]")
    print("   ", [short(d) for d in faiss_store.similarity_search(QUESTION, k=3)])

    print("\n  [Chroma — 거른 뒤 찾는다]")
    print("   ", [short(d) for d in chroma.similarity_search(QUESTION, k=3, filter=flt)])

    print("\n  [FAISS — 꺼낸 뒤 거른다]  fetch_k = 먼저 꺼내 올 개수")
    for fetch_k in (20, 4, 2):
        docs = faiss_store.similarity_search(QUESTION, k=3, filter=flt, fetch_k=fetch_k)
        print(f"    fetch_k={fetch_k:2d} → {len(docs)}개  {[short(d) for d in docs]}")

    print("""
  ★ 읽는 법
     · 결석 질문이라 가까운 조는 제5장(제29조)에 있습니다. 필터로 제7장만 보게 했습니다.
     · Chroma 는 DB 안에서 제7장만 남긴 뒤 찾으므로 항상 k 개를 채웁니다.
     · FAISS 는 가까운 순서로 fetch_k 개를 꺼낸 다음 파이썬에서 거릅니다.
       fetch_k 가 작으면 꺼낸 것이 전부 제5장이라 **k 보다 적게, 심하면 0개**가 나옵니다 ⚠️
     → 인터페이스는 같아도 동작은 다릅니다. 필터 문법도 저장소마다 다릅니다.

  여러 조건 (Chroma 문법)
     filter={"$and": [{"chapter": {"$eq": "..."}}, {"year": {"$gte": 2025}}]}
""")


def main() -> None:
    if "--scores" in sys.argv:
        demo_scores()
    elif "--filter" in sys.argv:
        demo_filter()
    else:
        demo_identity()


if __name__ == "__main__":
    main()
