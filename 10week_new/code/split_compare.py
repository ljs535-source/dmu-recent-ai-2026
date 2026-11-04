"""[3교시 실습 / 실습 2 — 2교시에 코드 읽기] ★★ 같은 문서를 3가지로 잘라, 같은 질문에 무엇이 검색되는지 비교한다

  ★ 이 실습이 10주차의 핵심입니다.
    보통 "임베딩 모델이 좋으면 검색이 잘 되겠지" 라고 생각하기 쉽습니다.
    실제로는 **어떻게 잘랐는가가 검색 품질의 상한을 결정합니다.**

    A 고정 크기   단순·빠름            ⚠️ 문장·단어 중간에서 끊긴다
    B 재귀적      큰 경계부터 시도      ★ 실무 기본값
    C 구조 기반   섹션 통째로 유지      ★★ 메타데이터가 자동 생성된다

  📌 결론 문장
     **"검색 품질의 상한은 임베딩 모델이 아니라 분할이 정합니다."**
     같은 임베딩 모델, 같은 질문인데 결과가 달랐습니다. 바꾼 것은 자르는 방법뿐입니다.

  이 파일은 3절(메타데이터 설계)의 build_chunks_with_metadata() 도 제공합니다.
  embed.py 가 이것을 가져다 씁니다. ★

실행:
    python split_compare.py            # 1단계: 어디서 끊겼는지 눈으로 본다 (무료·즉시) ★
    python split_compare.py --search   # 2단계: 세 인덱스에 같은 질문을 던진다 (임베딩 필요)
    python split_compare.py --meta     # 3절: 메타데이터를 심은 청크를 확인한다 ★
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HERE = Path(__file__).parent
DOC_PATH = HERE / os.getenv("WEEK10_DOC", "data/학칙.md")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")  # ≒0.6GB — 8GB GPU 기본값
# 더 큰 모델: "qwen3-embedding:4b"(≒2.5GB) · "qwen3-embedding:8b"(≒4.7GB, ❌ gemma3:4b 와 함께면 8GB 초과)
#            "bge-m3"(≒1.2GB, 다국어) — 자세한 비교는 embed.py 위쪽 주석

# ★ 세 인덱스에 똑같이 던질 질문 — .env 의 WEEK10_QUESTION 으로 바꿀 수 있습니다.
QUESTION = os.getenv("WEEK10_QUESTION", "몇 주를 결석하면 제적되나요?")
# 답 판정 — 검색된 청크에 이 문구가 들어 있으면 «답 포함» (제29조 2호)
ANSWER = os.getenv("WEEK10_ANSWER", "4주를 초과")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ★ 한국어 구분자를 추가했습니다.
#   기본 separators 는 영어 기준(". ")이라 한국어 문장 끝("다. ")을 못 잡습니다.
KO_SEPARATORS = ["\n\n", "\n", "다. ", ". ", " ", ""]

HEADERS = [("#", "장"), ("##", "조")]


def load_raw() -> str:
    return TextLoader(str(DOC_PATH), encoding="utf-8").load()[0].page_content


# ── A. 고정 크기 ────────────────────────────────────────────
def split_a(raw: str) -> list[str]:
    """500자에서 기계적으로 끊는다. ⚠️ 이 방식의 실패가 오늘 실습의 출발점."""
    return CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=0, separator=""
    ).split_text(raw)


# ── B. 재귀적 ──────────────────────────────────────────────
def split_b(raw: str) -> list[str]:
    """가능한 한 큰 경계를 지키면서 500자를 맞춘다. ★ 실무 기본값

       ① "\\n\\n"(문단)으로 나눠본다  → 500자 이하가 되면 채택
       ② 안 되면 "\\n"(줄)로
       ③ 안 되면 "다. "·". "(문장)으로
       ④ 그래도 안 되면 " "(단어) → 마지막엔 글자 단위
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=KO_SEPARATORS
    ).split_text(raw)


# ── C. 구조 기반 ───────────────────────────────────────────
def split_c(raw: str) -> list[Document]:
    """섹션(조)이 통째로 유지되고, ★★ metadata 가 자동으로 붙는다."""
    return MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS).split_text(raw)


# ── 청크 지도 — 세 방식을 «조 번호» 라는 같은 잣대로 본다 ★ ──────────
#   A·B·C 는 자르는 자리가 달라서 청크 본문을 그대로 나란히 놓으면 비교가 안 됩니다.
#   그래서 각 청크를 "어느 조가 들어 있는가" 로 바꿔 적습니다.
#   예) "…27·28·29·37" = 제27조 중간부터 제37조 끝까지 (… = 조 중간에서 잘림)
def article_spans(raw: str) -> list[tuple[int, int, str]]:
    """원문에서 조(##)마다 (시작, 끝, 번호) 를 구한다. 장(#) 제목은 조의 끝으로 본다."""
    heads, pos = [], 0
    for line in raw.splitlines(keepends=True):
        if line.startswith("#"):
            heads.append((pos, line))
        pos += len(line)
    spans = []
    for i, (start, line) in enumerate(heads):
        if line.startswith("## "):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(raw)
            no = line[3:].split("(")[0].removeprefix("제").replace("조", "", 1)  # "제42조의2" → "42의2"
            spans.append((start, end, no))
    return spans


def chunk_label(raw: str, chunk: str, spans, cursor: int = 0) -> tuple[str, int]:
    """청크 하나 → "…27·28·29·37…" 꼴 이름. (다음 찾기 위치도 돌려준다)"""
    s = raw.find(chunk[:30], cursor)
    if s < 0:
        return "?", cursor
    e = raw.find(chunk[-30:], s) + len(chunk[-30:])
    # 조 제목 줄만 딸려 온 경우(B 의 "… ## 제26조(복학)")는 그 조가 들어 있다고 보지 않는다
    hit = [(a, b, no) for a, b, no in spans
           if a < e and b > s and raw[max(raw.find("\n", a) + 1, s):min(b, e)].strip()]
    if not hit:
        return "(장 제목)", s + 1
    a0, _, _ = hit[0]
    _, b1, _ = hit[-1]
    # 조 제목 줄을 지나서 시작했으면 «중간부터», 조 끝까지 못 갔으면 «중간까지»
    head_end = raw.find("\n", a0) + 1
    cut_front = raw[head_end:s].strip() != ""
    cut_back = raw[e:b1].strip() != ""
    label = "·".join(no for _, _, no in hit)
    return ("…" if cut_front else "") + label + ("…" if cut_back else ""), s + 1


def chunk_labels(raw: str, name: str, chunks: list[str], c_docs=None) -> list[str]:
    if c_docs is not None:  # C 는 metadata 에 조가 이미 들어 있다 ★★
        return [d.metadata.get("조", "(장 제목)").split("(")[0].removeprefix("제").replace("조", "", 1)
                for d in c_docs]
    spans, cur, out = article_spans(raw), 0, []
    for ch in chunks:
        label, cur = chunk_label(raw, ch, spans, cur)
        out.append(label)
    return out


def compare_shapes(raw: str) -> dict:
    """먼저 눈으로 봅니다 — 어디서 끊겼는가 ★"""
    a, b = split_a(raw), split_b(raw)
    c_docs = split_c(raw)
    c = [d.page_content for d in c_docs]

    for name, chunks in [("A 고정", a), ("B 재귀", b), ("C 구조", c)]:
        lens = [len(x) for x in chunks]
        print("=" * 60)
        print(f"[{name}] {len(chunks)}개  평균 {sum(lens) // len(lens)}자  "
              f"최소 {min(lens)}  최대 {max(lens)}")
        print("--- 첫 청크 끝부분 80자 (어디서 끊겼는지 보기) ★ ---")
        print("..." + chunks[0][-80:].replace("\n", " "))

    print("=" * 60)
    print("""
  [A 고정] ...복학이 가능한 자는 금번 학     ← ⚠️ 단어 중간에서 끊겼다 («학|기»)
  [B 재귀] ...복학할 수 있다. ## 제26조(복학) ← ✅ 문단 경계 (다음 조 제목이 딸려 옴)
  [C 구조] (제25조 전체가 한 청크)          ← ✅ 조 단위

  ⚠️ A 에서 무슨 일이 일어났는지 보십시오.
     A 는 500자마다 기계적으로 자릅니다. 그래서
     ① "금번 학" | "기에 복학할 수 있다" 처럼 단어 중간이 끊기고
     ② 한 청크에 여러 조(전과 · 자퇴 · 제적)와 다른 장(평가시험)까지 섞입니다.
     → --search 에서 섞인 청크는 질문과 덜 닮아 보여, 답이 든 청크가 1위를 놓칩니다.
""")

    # C 의 metadata 가 자동 생성되는 것을 보여준다 ★★
    print("  ★★ C 는 metadata 가 자동으로 붙습니다 — 3절의 근거입니다")
    for d in c_docs[3:5]:
        print(f"    {d.metadata}")
    print()

    # ★ 청크 지도 — 본문 대신 «조 번호» 로 나란히 놓는다
    labels = {
        "A 고정": chunk_labels(raw, "A", a),
        "B 재귀": chunk_labels(raw, "B", b),
        "C 구조": chunk_labels(raw, "C", c, c_docs),
    }
    print("  청크 지도 — 청크마다 어느 조가 들어 있나 ★  (숫자 = 조 번호, … = 조 중간에서 잘림)")
    for name, ls in labels.items():
        print(f"    [{name}] " + " | ".join(ls))
    print("""
    A: 조 중간에서 잘리고(…), 한 청크에 여러 조가 섞인다 (예: …27·28·29·37)
    B: 조 경계에서 잘린다 (작은 조는 둘씩 묶임).  C: 청크 하나 = 조 하나
""")
    COMPARE_LABELS.update(labels)

    return {"A 고정": a, "B 재귀": b, "C 구조": c}


COMPARE_LABELS: dict[str, list[str]] = {}  # compare_search 가 같은 이름표를 쓴다


def compare_search(groups: dict) -> None:
    """세 인덱스에 같은 질문을 던져 무엇이 나오는지 본다 ★★"""
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    emb = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"\n질문: {QUESTION}\n")
    summary = []
    for name, chunks in groups.items():
        store = FAISS.from_texts(chunks, emb)
        hits = store.similarity_search(QUESTION, k=2)
        labels = dict(zip(chunks, COMPARE_LABELS.get(name, [])))
        print("=" * 60)
        print(f"[{name}]")
        for i, d in enumerate(hits, 1):
            text = d.page_content.replace("\n", " ")
            # ★ 답(숫자)이 이 청크 안에 실제로 들어 있는가 — 이것이 판정 기준
            has_answer = ANSWER in d.page_content
            mark = "✅ 답 포함" if has_answer else "❌ 답 없음"
            print(f"  {i}. [{mark}] 〔조 {labels.get(d.page_content, '?')}〕 {text[:110]}...")

        # ★ 답이 든 청크는 전체에서 몇 위였나 — 세 방식을 같은 잣대로 비교
        ranked = [d.page_content for d in store.similarity_search(QUESTION, k=len(chunks))]
        pos = next((i for i, t in enumerate(ranked, 1) if ANSWER in t), None)
        lens = [len(x) for x in chunks]
        summary.append((name, len(chunks), sum(lens) // len(lens),
                        f"{pos}위" if pos else "없음",
                        labels.get(ranked[pos - 1], "?") if pos else "-"))

    print("=" * 60)
    print(f"\n한눈에 비교 — 같은 질문 · 같은 잣대(답 «{ANSWER}» 이 든 청크) ★\n")
    print("  분할     청크 수  평균 길이  답 든 청크 순위  답 든 청크에 든 조")
    for name, n, avg, rank, label in summary:
        print(f"  {name}  {n:>6}  {avg:>8}자  {rank:>12}     {label}")
    print("\n  → 답 든 청크에 조가 많이 섞일수록 질문과 덜 닮아 보여 순위가 밀립니다. ★\n")
    print("=" * 60)
    print("""
학생이 채울 표

  | 분할   | 청크 수 | 평균 길이 | 검색 1위 청크에 **답이 들어 있는가** |
  | A 고정 |        |          |                                    |
  | B 재귀 |        |          |                                    |
  | C 구조 |        |          |                                    |

무엇을 읽어낼 것인가 ★

  A 에서 답이 없는 청크가 1위   답이 든 청크는 여러 조가 섞여 **2위로 밀린다** ★★
  B 는 문장이 온전함            같은 크기인데 **쓸 수 있는 청크**가 된다
  C 는 조 단위로 완결           규정·매뉴얼처럼 구조가 있는 문서에 최적
  C 의 청크 길이 편차가 큼       긴 섹션은 다시 잘라야 함 (2단 구성)

📌 **"검색 품질의 상한은 임베딩 모델이 아니라 분할이 정합니다."**

💡 질문을 바꿔 보세요 (.env 의 WEEK10_QUESTION). 표현에 따라 세 결과가 같아지기도, 더 갈리기도 합니다.
   chunk_size 를 바꿔도 결과가 달라집니다 — 무엇이 바뀌는지 직접 확인해 보세요. ★
""")


# ── 3절. 메타데이터 설계 ★★ ─────────────────────────────────
def build_chunks_with_metadata(raw: str | None = None) -> list[Document]:
    """분할 단계에서 메타데이터를 심는다. ★★ 이 절이 11주차 전부를 좌우합니다.

       [10주차 — 지금]                  [11주차 — 나중]
       청크에 metadata 를 심는다   →    · 필터 검색  "2026년 개정본에서만"
         source / page / 장 / 조         · Self-Query "5장에서만 찾아줘"
         year / category                 · ★ 출처 표기 "[학칙 제25조 ③]"
                                           │
       ⚠️ 안 심으면?                →     └ 만들 수 없습니다. 정보가 없으니까 ★★

    ★ 실무의 2단 구성을 씁니다: C(구조)로 크게 나눈 뒤 → 긴 섹션만 B(재귀)로 다시.
      · 구조로 나누면 장·조 metadata 가 공짜로 생깁니다
      · 재귀로 다듬으면 청크 길이 편차가 잡힙니다
    """
    raw = raw if raw is not None else load_raw()

    sections = split_c(raw)  # 장·조 metadata 가 붙은 섹션들
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
                        # ── 출처 표기용 ★ (11주차 3교시 Citation) ────────
                        "source": DOC_PATH.name,
                        "chapter": sec.metadata.get("장", ""),
                        "article": sec.metadata.get("조", ""),
                        # ⚠️ page 는 PDF Loader 라야 붙습니다. .md 에는 아예 없습니다.
                        #    "나중에 복원할 수 없는 것" 의 대표 사례입니다 ★
                        #    → 11주차 citation.py 에서 p.? 로 표시됩니다.
                        #    ⚠️ 값이 없다고 None 을 넣지 마십시오. 일부 벡터 저장소
                        #       (Chroma 등)는 metadata 에 None 을 거부합니다.
                        #       "없는 키" 로 두는 편이 안전합니다.
                        # ── 필터 검색용 ★ (11주차 1교시) ────────────────
                        # ⚠️ 타입을 맞추십시오. 2026(숫자)과 "2026"(문자)은
                        #    범위 필터에서 다르게 동작합니다.
                        "year": 2026,
                        "category": "학사",
                        "doc_type": "규정",
                        # ── 운영·디버깅용 ────────────────────────────────
                        "chunk_id": len(chunks),
                        # ★ 어떻게 잘랐는지 기록해 두면 7주차 방식으로
                        #   분할 전략을 A/B 비교할 수 있습니다.
                        "splitter": f"markdown→recursive-{CHUNK_SIZE}-{CHUNK_OVERLAP}",
                    },
                )
            )
    return chunks


def show_metadata() -> None:
    chunks = build_chunks_with_metadata()
    print(f"메타데이터를 심은 청크 {len(chunks)}개\n")
    for d in chunks[:3]:
        print("─" * 60)
        print(d.metadata)
        print(d.page_content[:90].replace("\n", " "), "...")

    print("─" * 60)
    print("""
설계 원칙 3가지 ★

  나중에 복원할 수 없는 것을 우선   페이지 번호·섹션 제목은 자르고 나면 사라진다 ★
  필터로 쓸 값은 타입을 맞춘다      year: 2026(숫자) vs "2026"(문자) — 범위 필터가 달라짐
  본문에 이미 있는 것은 안 넣는다    중복은 저장 비용만 늘림

⚠️ 과하게도, 부족하게도 넣지 마십시오. 기준은 하나입니다 —
   ***"11주차에 이걸로 무엇을 하고 싶은가?"***

📌 해 보기: "내 미니 프로젝트 문서에는 어떤 메타데이터가 필요할까?" 를
   CONTEXT.md 에 한 줄이라도 적어 보세요. ★
""")


def main() -> None:
    raw = load_raw()
    print(f"문서: {DOC_PATH}  ({len(raw)}자)\n")

    if "--meta" in sys.argv:
        show_metadata()
        return

    groups = compare_shapes(raw)

    if "--search" in sys.argv:
        compare_search(groups)
    else:
        print("  다음: python split_compare.py --search   # 세 인덱스에 같은 질문 ★")
        print("        python split_compare.py --meta     # 3절 메타데이터 설계 ★★")


if __name__ == "__main__":
    main()
