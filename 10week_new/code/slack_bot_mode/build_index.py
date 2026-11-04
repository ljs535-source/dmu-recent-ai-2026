"""[10주차 · 슬랙 연동 준비 ①] 학칙 PDF 로 봇 전용 인덱스를 만든다

  10주차 3교시 embed.py 는 학칙.md (제5·7장 정리본, 3,388자) 로 인덱스를 만들었다.
  봇은 **학칙 전체 PDF** (28쪽, 약 2.7만 자) 를 근거로 답한다.

    data/학칙.pdf ─▶ ① 적재   PyPDFLoader — 1쪽 = Document 1개 (1교시)
                  ─▶ ② 정리   머리말 · 쪽 번호 제거, 줄 끝에서 끊긴 단어 잇기 ★
                  ─▶ ③ 분할   «제N조» 로 먼저 나누고 → 긴 조만 재귀로 다시 (2교시 2단 구성)
                  ─▶ ④ 메타   source · page · chapter · article 을 심는다 (2교시 2절) ★★
                  ─▶ ⑤ 임베딩 + 저장   slack_bot_mode/index_pdf/  (3교시 실습 3)

★ 1교시에서 말한 «PDF 파싱 난점» 이 이 파일에 전부 들어 있다
  - 쪽마다 머리말    «동양미래대학교 규정집 제2편 학칙   2-0-1~9»  와  «- 9 -»
    → 지우지 않으면 모든 청크에 같은 문장이 섞여 검색을 흐린다
  - 줄 끝에서 끊긴 단어  «이\\n를 취소한다»
    → 그대로 두면 '이' 와 '를' 이 다른 낱말이 된다
  - 표 (입학정원 · 성적 등급)  → 숫자 나열로 풀린다. 이 파일은 고치지 않는다 (범위 밖)

⚠️ ../index_recursive/ (embed.py 결과) 는 건드리지 않는다 — 11·13주차가 그대로 이어 쓴다.

실행 (이 폴더에서):
    python build_index.py              # data/학칙.pdf → index_pdf/
    python build_index.py --show       # 저장 없이 청크 몇 개와 metadata 만 본다 (임베딩 없음 · 즉시)
    python build_index.py 내문서.pdf    # 미니 프로젝트 — 다른 PDF 로 (조 구조가 없으면 쪽 단위로)
"""

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from langchain_community.document_loaders import PyPDFLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path.insert(0, str(CODE))

from faiss_io import load_index, save_index  # noqa: E402  ← 한글 경로 대비책 (3교시)
from split_compare import CHUNK_OVERLAP, CHUNK_SIZE, KO_SEPARATORS  # noqa: E402  ← 10주차 분할 설정 그대로

PDF_PATH = CODE / "data" / "학칙.pdf"
PDF_URL = "https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do"  # ⚠️ PDF 는 저장소에 없다 — 여기서 받는다
INDEX_DIR = HERE / "index_pdf"          # ★ 봇 전용 — rag_backend.py 가 읽는다
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

# ── ② 정리 규칙 ───────────────────────────────────────────────────
HEADER = re.compile(r"^동양미래대학교 규정집.*$|^- \d+ -$", re.M)      # 머리말 · 쪽 번호
# 이 모양으로 시작하는 줄은 '새 줄' 이다 — 앞 줄에 붙이지 않는다
NEW_LINE = re.compile(r"^(제\s*\d+\s*장|제\d+조|부\s*칙|\[별|\s|[①-⑳]|\d+\.|<)")
CHAPTER = re.compile(r"^제\s*(\d+)\s*장\s+(.+)$")                       # «제 5 장  휴학, 퇴학 …»
APPENDIX = re.compile(r"^부\s*칙")                                       # «부   칙» — 개정 때마다 붙는 시행일 조항
ANNEX = re.compile(r"^(\[별[표지][^\]]*\])")                              # «[별표 1]» · «[별지 제1호서식]» — 23쪽부터
ARTICLE = re.compile(r"^(제\d+조(?:의\d+)?\([^)]*\))")                   # «제29조(제적)» · «제1조의2(교육목표)»


def chapter_name(no: str, name: str) -> str:
    """«총  칙» · «학  생  활  동» 처럼 글자 사이를 벌린 제목을 붙인다."""
    words = name.split()
    name = "".join(words) if all(len(w) == 1 for w in words) else " ".join(words)
    return f"제{no}장 {name}"


def clean(text: str) -> str:
    """머리말을 지우고, 줄 끝에서 끊긴 문장을 잇는다."""
    text = HEADER.sub("", text)
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if out and not NEW_LINE.match(line):
            out[-1] += line          # ★ «이» + «를 취소한다» → «이를 취소한다»
        else:
            out.append(line)
    return "\n".join(out)


# ── ③ 구조로 먼저 나눈다 — «제N조» 마다 한 덩어리 ──────────────────
def split_articles(pages: list[Document]) -> list[Document]:
    """쪽을 넘나들며 조 단위 섹션을 만든다. 조가 시작한 쪽 번호를 기억한다 ★"""
    sections: list[Document] = []
    chapter, article, page, buf = "", "", 1, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            meta = {"source": PDF_PATH.name, "page": page, "chapter": chapter, "article": article}
            # ⚠️ 값이 없는 키는 넣지 않는다 (2교시 — None/빈 값을 싫어하는 저장소가 있다)
            sections.append(Document(page_content=body, metadata={k: v for k, v in meta.items() if v != ""}))

    for doc in pages:
        page_no = doc.metadata.get("page", 0) + 1          # PyPDFLoader 는 0 부터 센다
        for line in clean(doc.page_content).splitlines():
            if m := CHAPTER.match(line.strip()):
                flush(); buf = []
                chapter, article, page = chapter_name(*m.groups()), "", page_no
                continue
            if APPENDIX.match(line.strip()):
                # ⚠️ 부칙은 조 번호가 «제1조» 부터 다시 시작한다 — 본문 제1조와 헷갈리지 않게 표시
                flush(); buf = []
                chapter, article, page = "부칙", "", page_no
                continue
            if m := ANNEX.match(line.strip()):
                flush(); buf = []
                chapter, article, page = m.group(1), "", page_no
            elif m := ARTICLE.match(line.strip()):
                flush(); buf = []
                article = ("부칙 " if chapter == "부칙" else "") + m.group(1)
                page = page_no
            elif not buf:
                page = page_no
            buf.append(line.strip())
    flush()
    return sections


def build_chunks(pdf_path: Path = PDF_PATH) -> list[Document]:
    """① 적재 → ② 정리 → ③ 구조 분할 → 긴 섹션만 재귀 분할 → ④ 메타데이터"""
    pages = PyPDFLoader(str(pdf_path)).load()                                   # ①
    sections = split_articles(pages)                                            # ②③
    if len(sections) <= 1:  # 조 구조가 없는 문서 — 쪽 단위로
        sections = [Document(page_content=clean(p.page_content),
                             metadata={"source": pdf_path.name, "page": p.metadata.get("page", 0) + 1})
                    for p in pages]

    sub = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                         separators=KO_SEPARATORS)
    chunks: list[Document] = []
    for sec in sections:
        for piece in sub.split_text(sec.page_content):
            chunks.append(Document(page_content=piece, metadata={
                **sec.metadata,                                                 # ④
                "chunk_id": len(chunks),
                "splitter": f"pdf-article→recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}",
            }))
    return chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pdf = Path(args[0]).resolve() if args else PDF_PATH
    if not pdf.exists():
        raise SystemExit(f"PDF 가 없습니다: {pdf}\n"
                         f"  → 학교 규정집에서 받아 data/학칙.pdf 로 넣으세요:\n     {PDF_URL}")

    chunks = build_chunks(pdf)
    lengths = [len(c.page_content) for c in chunks]
    print(f"PDF     : {pdf.name}")
    print(f"청크    : {len(chunks)}개  (평균 {sum(lengths) // len(lengths)}자 · 최대 {max(lengths)}자)")
    print(f"조 수   : {len({c.metadata.get('article') for c in chunks if c.metadata.get('article')})}개")

    if "--show" in sys.argv:
        for c in chunks:
            if c.metadata.get("article", "").startswith(("제25조(", "제29조(")):
                print("\n", c.metadata)
                print("   ", c.page_content[:150].replace("\n", " / "), "…")
        return

    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    print(f"임베딩  : {EMBED_MODEL} — 잠시 걸립니다…")
    store = FAISS.from_documents(chunks, emb)
    how = save_index(store, INDEX_DIR)
    print(f"✅ 저장 : {INDEX_DIR}  (방식: {how})")

    hit = load_index(INDEX_DIR, emb).similarity_search("몇 주를 결석하면 제적되나요?", k=1)[0]
    print(f"   되읽기 확인: {hit.metadata.get('article')} p.{hit.metadata.get('page')} | "
          f"{hit.page_content[:50].replace(chr(10), ' ')}…")
    print("\n다음:  python rag_backend.py 검색   →   python slack_rag_bot.py")


if __name__ == "__main__":
    main()
