"""[미니 프로젝트] 내 데이터 → 봇이 읽는 인덱스  (my_data/ → index_my/)

   10주차에 배운 RAG 준비 다섯 단계를 내 문서에 그대로 합니다.

     my_data/*.pdf · *.md · *.txt · *.csv
        ─▶ ① 적재   PDF 는 쪽마다 · MD 는 «## 제목» 마다 · CSV 는 행마다 Document 1개     (10주차 1교시)
        ─▶ ② 분할   500자 · 50자 겹침 · 한국어 구분자 — 10주차 split_compare.py 와 같은 설정  (10주차 2교시)
        ─▶ ③ 메타   source(파일 이름) · page(PDF 쪽) · row(CSV 행) · title(제목) · chunk_id    (10주차 2교시 2절)
        ─▶ ④ 임베딩  EMBED_MODEL (기본 qwen3-embedding:0.6b)                                    (10주차 3교시)
        ─▶ ⑤ 저장   index_my/  — faiss_io.save_index (경로에 한글이 있어도 된다)

   ★ ③ 메타가 슬랙 답 아래 «📚 근거  회칙.md · 행3» 로 나옵니다 (bot_config.source_label).
   ★ CSV 는 첫 줄이 열 이름이어야 합니다. 한 행이 «열이름: 값» 줄들로 바뀌어 한 청크가 됩니다 (FAQ · 일정표에 알맞음).
     첫 열 값을 title 로 씁니다 — FAQ 라면 첫 열을 «질문» 으로 두세요.

실행 (이 폴더에서):
    python build_my_index.py --show            # 저장 없이 청크 수 · metadata 만 (임베딩 없음 · 즉시) ★ 먼저
    python build_my_index.py                   # my_data/ → index_my/
    python build_my_index.py 내폴더 --out index_club   # 다른 폴더 · 다른 인덱스 이름
그다음 bot_config.py 의 INDEX 를 "index_my" 로 바꾸거나, 루트 .env 에 BOT_INDEX=index_my 를 적습니다.
"""

from dotenv import load_dotenv

load_dotenv()

import csv  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # 13주차 code — faiss_io.py

from faiss_io import load_index, save_index  # noqa: E402

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")  # ⚠️ 봇이 읽을 때도 같은 모델
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50                               # 10주차 split_compare.py 와 같은 값
KO_SEPARATORS = ["\n\n", "\n", "다. ", ". ", " ", ""]
KINDS = {".pdf", ".md", ".txt", ".csv"}


# ── ① 적재 ────────────────────────────────────────────────────────
def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "cp949"):  # 윈도 메모장 · 엑셀에서 저장한 파일은 cp949 일 때가 있다
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"⚠️ 글자 인코딩을 읽지 못했습니다: {path.name} — UTF-8 로 다시 저장하세요.")


def first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("#"):
            return line.strip("# ").strip()[:60]
    return ""


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(path)).load()
        return [Document(p.page_content, metadata={"source": path.name, "page": p.metadata.get("page", 0) + 1})
                for p in pages if p.page_content.strip()]
    if ext in (".md", ".txt"):
        text = read_text(path)
        # ★ MD 는 «## 제목» 마다 나눈다 — 그 제목이 출처가 된다 (10주차 학칙을 «제N조» 로 먼저 나눈 것과 같은 생각)
        sections, title, buf = [], first_heading(text), []
        for line in text.splitlines():
            if line.startswith("## ") and buf:
                sections.append((title, "\n".join(buf)))
                buf = []
            if line.startswith("## "):
                title = line.strip("# ").strip()[:60]
            buf.append(line)
        sections.append((title, "\n".join(buf)))
        docs = []
        for t, body in sections:
            if body.strip():
                meta = {"source": path.name}
                if t:
                    meta["title"] = t
                docs.append(Document(body, metadata=meta))
        return docs
    if ext == ".csv":
        rows = list(csv.DictReader(read_text(path).splitlines()))
        docs = []
        for i, row in enumerate(rows, start=2):  # 1행은 열 이름 — 엑셀에서 보이는 행 번호와 맞춘다
            body = "\n".join(f"{k}: {v}" for k, v in row.items() if k and v and str(v).strip())
            if body:
                title = str(next(iter(row.values()), "") or "")[:60]
                docs.append(Document(body, metadata={"source": path.name, "row": i, "title": title}))
        return docs
    return []


def load_all(paths: list[Path]) -> list[Document]:
    files = []
    for p in paths:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in KINDS) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"⚠️ 읽을 파일이 없습니다: {', '.join(str(p) for p in paths)}\n"
                         f"   PDF · MD · TXT · CSV 파일을 my_data/ 에 넣으세요.")
    docs = []
    for f in files:
        got = load_file(f)
        print(f"  ① {f.name:30s} → Document {len(got)}개")
        docs += got
    return docs


# ── ② 분할 + ③ 메타 ───────────────────────────────────────────────
def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=KO_SEPARATORS)
    chunks = splitter.split_documents(docs)  # metadata(source · page · row · title)는 청크로 그대로 이어진다
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    out = "index_my"
    if "--out" in args:
        out = args[args.index("--out") + 1]
        del args[args.index("--out"):args.index("--out") + 2]
    paths = [Path(a).resolve() for a in args if not a.startswith("--")] or [HERE / "my_data"]
    index_dir = (HERE / out).resolve()

    print(f"입력 : {', '.join(p.name for p in paths)}   → 출력 : {index_dir.name}/   (임베딩 {EMBED_MODEL})")
    chunks = split(load_all(paths))
    lens = [len(c.page_content) for c in chunks]
    print(f"  ② 분할 → 청크 {len(chunks)}개 · 평균 {sum(lens) // max(len(lens), 1)}자 · 가장 긴 것 {max(lens, default=0)}자")

    if "--show" in sys.argv:
        print("\n── ③ 청크 몇 개와 metadata (저장 안 함) ──")
        for c in chunks[:: max(len(chunks) // 4, 1)][:4]:
            print(f"\n {c.metadata}\n    {c.page_content[:120].replace(chr(10), ' / ')} …")
        print("\n→ metadata 의 source · page · row · title 이 슬랙 답의 «📚 근거» 가 됩니다. 괜찮으면 --show 없이 다시.")
        return

    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    print(f"  ④ 임베딩 중… (청크 {len(chunks)}개 — GPU 가 없으면 몇 분 걸릴 수 있습니다)")
    store = FAISS.from_documents(chunks, emb)
    how = save_index(store, index_dir)
    print(f"  ⑤ 저장 ✅ {index_dir}  (방식: {how})")

    probe = chunks[0].page_content.split("\n")[0][:40]
    hit = load_index(index_dir, emb).similarity_search(probe, k=1)[0]
    print(f"\n  확인 — «{probe}» 로 검색하면 1위: {hit.metadata}")
    print(f"\n다음: bot_config.py 의 INDEX 를 \"{out}\" 로 (또는 루트 .env 에 BOT_INDEX={out}) → python bot_graph.py")


if __name__ == "__main__":
    main()
