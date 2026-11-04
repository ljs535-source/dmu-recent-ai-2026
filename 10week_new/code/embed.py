"""[3교시 실습 / 실습 3] 청크를 벡터로 바꾸고, 인덱스로 저장한다

  🎯 오늘은 "고르고 붙이는 것" 만 합니다. 임베딩 원리는 2주차에 이미 배웠습니다.
     다만 하나는 반드시 확인합니다 — **이 모델이 한국어를 제대로 다루는가?** ★★

  임베딩 모델 선택 기준 4가지 ★

     한국어 지원 ★★   한국어를 학습한 모델인가   영어 전용은 한국어 유사도가 무너진다
     차원              384 / 768 / 1024 …        크면 표현력↑ 저장·검색 비용↑
     속도              초당 처리 청크 수          문서 1만 개면 차이가 큽니다
     로컬 실행 가능    VRAM 요구량                8GB 안에 생성 LLM과 함께 올라가는가 ★

     ⚠️ 8GB VRAM 예산 (11주차 기준)
        생성 LLM (8B Q4)     ≒ 5.0 GB
        임베딩 모델          ≒ 0.3 GB
        KV 캐시 + 오버헤드   ≒ 1.5 GB
        ─────────────────────────────
        합계                 ≒ 6.8 GB   → 여유 있음 ✅

  ⚠️⚠️ 마지막 단계(인덱스 저장)를 반드시 하고 커밋하세요.
       **11주차 실습 1과 13주차 실습 4가 오늘 만든 인덱스를 그대로 이어 씁니다.**
       없으면 다음 주 도입부를 재적재로 날립니다.

실행:
    python embed.py --check     # 한국어가 제대로 되는지 검증 (2주차 코사인 유사도) ★★
    python embed.py             # 청크 임베딩 + 인덱스 저장 ★★
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama import OllamaEmbeddings

from faiss_io import load_index, save_index  # ⚠️ 한글 경로 대비책 ★★
from split_compare import build_chunks_with_metadata  # ★ 2교시 2절 메타데이터 설계를 실습 2 --meta 에서 만든 것

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")  # ≒0.6GB · 1024차원 — 8GB GPU 기본값
# 더 큰 모델 (8GB GPU 에 생성 모델 gemma3:4b ≒3.3GB 와 함께 올린다고 보고 고르십시오):
#   "qwen3-embedding:4b"  ≒2.5GB · 2560차원  — 8GB 에서 여유 있음
#   "qwen3-embedding:8b"  ≒4.7GB · 4096차원  — ❌ gemma3:4b + KV 캐시와 합치면 ≒9.5GB, 8GB 초과
#   "bge-m3"              ≒1.2GB · 1024차원  — 다국어(한국어 포함)
# ⚠️ 모델을 바꾸면 차원이 달라지므로 인덱스를 다시 만들어야 합니다 (11·13주차 EMBED_MODEL 도 같게).

# ⚠️ 이 경로 이름을 바꾸지 마십시오. 11·13주차 코드가 이 경로를 그대로 씁니다. ★★
INDEX_DIR = Path(__file__).parent / "index_recursive"


def basics(emb: OllamaEmbeddings) -> None:
    """① 한 문장 임베딩 / ② 청크 여러 개를 한 번에"""
    print("── ① 한 문장 임베딩 ────────────────────────────")
    v = emb.embed_query("휴학기간은 최대 얼마까지 가능한가?")
    print("  차원   :", len(v))
    print("  앞 5개 :", [round(x, 4) for x in v[:5]])

    print("""
  ★ embed_query 와 embed_documents 를 구분하십시오.
    일부 임베딩 모델은 **질문과 문서를 다르게 처리**합니다(접두어를 붙이는 등).
    그래서 메서드가 나뉘어 있습니다. 섞어 쓰면 검색 품질이 떨어질 수 있습니다.
""")


def korean_check(emb: OllamaEmbeddings, model_name: str = "") -> bool:
    """②-2. 이 모델이 한국어 '의미' 를 잡습니까? ★★

    2주차에 배운 코사인 유사도가 여기서 **검증 도구**로 쓰입니다.

    ★★ 판정 기준은 '점수의 절대값' 이 아닙니다.
       **의미가 같은 쌍이 의미가 다른 쌍보다 확실히 높은가** 입니다.
       (임베딩 모델마다 점수 분포가 달라서 절대값은 비교 기준이 못 됩니다)
    """
    import numpy as np

    def cos(a, b) -> float:
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # (질문, 비교문, 설명, 같은 의미인가)
    pairs = [
        ("휴학 절차", "학교를 잠시 쉬는 방법", "높아야 함 ★", True),
        ("휴학 절차", "졸업 요건", "낮아야 함 ★", False),
        ("휴학 절차", "휴학 신청 방법", "매우 높아야 함", True),
        ("졸업 학점", "졸업에 필요한 학점 수", "높아야 함", True),
        ("졸업 학점", "군 복무 휴학 신청", "낮아야 함", False),
    ]

    print(f"── 한국어 검증 ({model_name or EMBED_MODEL}) ★★ ──────────")
    same, diff = [], []
    for a, b, expect, is_same in pairs:
        s = cos(emb.embed_query(a), emb.embed_query(b))
        (same if is_same else diff).append(s)
        print(f"  {s:.3f}  {a} ↔ {b}   ({expect})")

    # ★ 핵심 지표: '의미가 같은 쌍' 의 최저점 − '의미가 다른 쌍' 의 최고점
    #   이 값이 0 이하면 두 무리가 **겹칩니다** = 의미를 구분하지 못합니다.
    margin = min(same) - max(diff)
    print(f"\n  같은 의미 최저 {min(same):.3f}  ↔  다른 의미 최고 {max(diff):.3f}")
    print(f"  ★ 판정 지표(margin) = {margin:+.3f}")

    ok = margin > 0.03
    if ok:
        print("""
  ✅ 의미가 같은 쌍이 확실히 높습니다. 이 모델을 써도 됩니다.
""")
    else:
        print(f"""
  ⚠️⚠️ **이 모델은 한국어 의미를 구분하지 못합니다.** (margin {margin:+.3f} ≤ 0.03)

     "휴학 절차" 와 "졸업 요건"(전혀 다른 내용)의 점수가
     "휴학 절차" 와 "학교를 잠시 쉬는 방법"(같은 내용)만큼 높습니다.
     **점수가 높은 것이 문제가 아니라, 구분이 안 되는 것이 문제입니다.** ★★

     ⚠️ 그대로 두면 11주차에 검색이 안 되는데 **원인을 못 찾게 됩니다.**
        (검색 전략 3종을 비교해도 전부 잡음이 나옵니다)

  💡 **교체를 검토하세요.** 한국어를 학습한 임베딩 모델 후보:
        ollama pull bge-m3                 # 다국어(한국어 포함) · 1024차원 · ~1.2GB
        ollama pull qwen3-embedding:0.6b   # 다국어 · 소형
     그리고 .env 의 EMBED_MODEL 을 바꾼 뒤 이 검증을 **다시** 돌리십시오:
        python embed.py --check {EMBED_MODEL}
        python embed.py --check bge-m3
""")
    return ok


def build_and_save(emb: OllamaEmbeddings) -> None:
    """③ 산출물 저장 — 11주차에 그대로 이어 씁니다 ★★"""
    from langchain_community.vectorstores import FAISS

    # ⚠️ chunks 는 실습 2 --meta 에서 만든 'Document 리스트' 입니다.
    #    split_text() 가 돌려준 '문자열 리스트' 가 아닙니다 —
    #    메타데이터가 붙어 있어야 11주차 필터 검색·출처 표기가 가능합니다. ★
    chunks = build_chunks_with_metadata()
    texts = [c.page_content for c in chunks]

    print("── ② 청크 여러 개를 한 번에 ★ ──────────────────")
    vectors = emb.embed_documents(texts)
    print(f"  {len(vectors)}개 청크 → {len(vectors[0])}차원 벡터\n")

    store = FAISS.from_documents(chunks, emb)  # ★ 메타데이터까지 함께 들어간다

    # ⚠️ 슬라이드 코드는 store.save_local(...) 한 줄입니다.
    #    경로에 한글이 있으면 그 한 줄이 실패합니다 — faiss_io.py 의 설명을 보십시오. ★★
    how = save_index(store, INDEX_DIR)

    print(f"✅ 인덱스 저장: {INDEX_DIR}  (방식: {how})")
    print(f"   파일: {sorted(f.name for f in INDEX_DIR.iterdir())}")

    # 저장한 것이 실제로 읽히는지 그 자리에서 확인 ★
    reloaded = load_index(INDEX_DIR, emb)
    hit = reloaded.similarity_search("몇 주를 결석하면 제적되나요?", k=1)[0]
    print(f"   되읽기 확인: {hit.metadata.get('article')} | "
          f"{hit.page_content[:50].replace(chr(10), ' ')}...")

    print("""
⚠️⚠️ 반드시 커밋하십시오.
   **11주차 실습 1**과 **13주차 실습 4**가 이 인덱스를 그대로 이어 씁니다.
   없으면 다음 주 도입부를 재적재로 날립니다.

       git add index_recursive
       git commit -m "week10: 인덱스 저장"

""")


def main() -> None:
    if "--check" in sys.argv:
        # ★ 모델을 인자로 여러 개 넘기면 후보들을 나란히 비교합니다.
        #   python embed.py --check qwen3-embedding:0.6b nomic-embed-text
        names = [a for a in sys.argv[1:] if not a.startswith("-")] or [EMBED_MODEL]
        results = {}
        for name in names:
            try:
                results[name] = korean_check(OllamaEmbeddings(model=name), name)
            except Exception as e:
                print(f"── {name}: ❌ 사용 불가 ({type(e).__name__}) — ollama pull 이 필요합니다\n")
                results[name] = False
        if len(results) > 1:
            print("=" * 60)
            for name, ok in results.items():
                print(f"  {'✅' if ok else '❌'}  {name}")
            print("\n  ✅ 인 모델을 .env 의 EMBED_MODEL 에 적으세요. ★")
        return

    print(f"임베딩 모델: {EMBED_MODEL}\n")
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    basics(emb)
    if not korean_check(emb):
        print("""⚠️ 한국어 검증에 실패했지만 인덱스는 만듭니다 (파이프라인 학습은 가능).
   다만 **11주차 검색 실습의 결과가 잡음이 됩니다.**
   임베딩 모델을 바꾼 뒤 인덱스를 다시 만드세요. ★★
""")
    print("=" * 60)
    build_and_save(emb)


if __name__ == "__main__":
    main()
