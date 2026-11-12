"""[공용] 10주차 인덱스를 이어 쓰기 위한 잡일 모음

  ⚠️ 11주차 실습은 전부 **10주차에 저장한 인덱스**에서 출발합니다.
        10week_new/code/index_recursive/   (실제 학칙 · qwen3-embedding:0.6b · 1024차원)

  이 파일이 하는 일
     · 인덱스가 있으면 → 그대로 읽는다 (빠름)
     · 없으면        → 10주차 학칙.md 로 그 자리에서 다시 만든다 (⏱ 1~2분)
     · 청크 목록도 함께 돌려준다 (2교시 하이브리드의 BM25 인덱스에 필요) ★

이 파일은 단독 실행용이 아닙니다. 다른 실습 파일들이 가져다 씁니다.
(단독으로 돌리면 인덱스 상태만 점검합니다 — 수업 시작 전에 한 번 돌려 보세요 ★)

실행:
    python rag_common.py        # 인덱스 · metadata 점검
"""

from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # ★ 반드시 맨 위 — 저장소 맨 위의 .env 를 찾아 읽습니다

from langchain_core.documents import Document

HERE = Path(__file__).parent
INDEX_DIR = (HERE / os.getenv("WEEK11_INDEX", "../../10week_new/code/index_recursive")).resolve()
DOC_PATH = (HERE / os.getenv("WEEK11_DOC", "../../10week_new/code/data/학칙.md")).resolve()

MODEL = os.getenv("MODEL", "gemma3:4b")
# ⚠️ 10주차 인덱스를 만든 모델과 같아야 합니다. 다르면 차원이 달라 검색이 안 됩니다.
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# 10주차와 **같은 분할 설정** 이어야 합니다. 다르면 비교가 흔들립니다. ★
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
KO_SEPARATORS = ["\n\n", "\n", "다. ", ". ", " ", ""]
HEADERS = [("#", "장"), ("##", "조")]


# ── 표를 그리기 위한 잡일 (7주차 metrics.py 와 같은 것) ──────
#    한글·★ 는 표시 폭이 2인데 len() 은 1로 셉니다.
#    그대로 f"{s:<20}" 하면 표의 세로줄이 어긋납니다.
def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s: str, n: int, align: str = "<") -> str:
    """표시 폭 기준으로 채운다. align: '<' 왼쪽 / '>' 오른쪽."""
    s = str(s)
    fill = " " * max(0, n - width(s))
    return s + fill if align == "<" else fill + s


def clip(s: str, n: int) -> str:
    """표시 폭 기준으로 자른다."""
    out = ""
    for c in str(s).replace("\n", " "):
        if width(out) + width(c) > n:
            break
        out += c
    return out


def get_llm(temperature: float = 0):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=MODEL, temperature=temperature)


def get_emb():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBED_MODEL)


def rebuild_chunks() -> list[Document]:
    """10주차 2교시 메타데이터 설계(split_compare.py --meta)를 그대로 재현한다.

    구조(C)로 장·조를 얻고 → 긴 조만 재귀(B)로 한 번 더 자른다.
    ⚠️ CHUNK_SIZE / CHUNK_OVERLAP / separators 를 바꾸면 10주차와 결과가 달라집니다.
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    raw = DOC_PATH.read_text(encoding="utf-8")
    sections = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS).split_text(raw)
    sub = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=KO_SEPARATORS
    )

    chunks: list[Document] = []
    for sec in sections:
        for piece in sub.split_text(sec.page_content):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": DOC_PATH.name,
                        "chapter": sec.metadata.get("장", ""),
                        "article": sec.metadata.get("조", ""),
                        "year": 2026,
                        "category": "학사",
                        "doc_type": "규정",
                        "chunk_id": len(chunks),
                        "splitter": f"markdown→recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}",
                    },
                )
            )
    return chunks


def load_store(verbose: bool = True):
    """FAISS 저장소를 돌려준다. 10주차 인덱스가 있으면 그대로, 없으면 다시 만든다."""
    from langchain_community.vectorstores import FAISS

    from faiss_io import index_exists, load_index, save_index  # ⚠️ 한글 경로 대비책 ★

    emb = get_emb()

    if index_exists(INDEX_DIR):
        if verbose:
            print(f"✅ 10주차 인덱스를 이어 씁니다: {INDEX_DIR}")
        return load_index(INDEX_DIR, emb)

    if verbose:
        print(f"""⚠️ 10주차 인덱스가 없습니다 ({INDEX_DIR})
   → {DOC_PATH.name} 으로 그 자리에서 다시 만듭니다. (⏱ 1~2분)""")

    if not DOC_PATH.exists():
        sys.exit(f"⚠️ 원본 문서도 없습니다: {DOC_PATH}\n"
                 f"   .env 의 WEEK11_INDEX / WEEK11_DOC 를 확인하세요.")

    store = FAISS.from_documents(rebuild_chunks(), emb)
    save_index(store, INDEX_DIR)
    if verbose:
        print(f"   저장했습니다: {INDEX_DIR}")
    return store


def load_chunks(store=None) -> list[Document]:
    """저장소에 들어 있는 청크 목록. ★ 2교시 BM25 인덱스에 필요합니다.

    BM25 는 '문서 텍스트' 를 따로 색인해야 합니다 — 벡터만으로는 못 만듭니다.
    """
    store = store or load_store(verbose=False)
    try:
        # FAISS 는 벡터만 갖고, 글과 metadata 는 바깥 docstore 에 따로 둡니다 ★
        return [store.docstore.search(i) for i in store.index_to_docstore_id.values()]
    except Exception:
        return rebuild_chunks()


def health_check() -> None:
    from faiss_io import index_exists, path_is_ascii

    print("── 10주차 인덱스 점검 ★ ────────────────────────")
    print(f"  인덱스 경로 : {INDEX_DIR}")
    print(f"  존재 여부   : {'✅ 있음' if index_exists(INDEX_DIR) else '❌ 없음'}")
    print(f"  원본 문서   : {DOC_PATH}  ({'✅' if DOC_PATH.exists() else '❌'})")
    print(f"  임베딩 모델 : {EMBED_MODEL}")
    if not path_is_ascii(INDEX_DIR):
        print("  💡 경로에 한글이 있습니다 — faiss_io.py 가 index.bytes 방식으로 읽습니다 (정상).")

    store = load_store()
    chunks = load_chunks(store)
    print(f"\n  청크 수     : {len(chunks)}")
    print(f"  벡터 차원   : {store.index.d}   ← 임베딩 모델이 정합니다")
    print(f"  FAISS 인덱스: {type(store.index).__name__}   ← LangChain 기본값 = 전부 비교 ★")
    if chunks:
        print(f"  metadata 예 : {chunks[0].metadata}")

    keys = set()
    for c in chunks:
        keys |= set(c.metadata)
    print(f"  metadata 키 : {sorted(keys)}")

    chapters = sorted({c.metadata.get("chapter", "") for c in chunks})
    print(f"  chapter 값  : {chapters}   ← 1교시 필터 검색에 씁니다 ★")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    health_check()
