"""[3교시 실습 / 실습 1 — 1교시에 코드 읽기] 문서 형식별 Loader 4종을 돌려보고 결과를 비교한다

  Loader = 무엇이든 읽어서 Document 객체 리스트로 만든다.

      Document(
          page_content="문서의 본문 텍스트...",
          metadata={"source": "학칙.pdf", "page": 7},      # ★ 이게 오늘의 복선
      )

  ★ metadata 를 지금 기억해 두십시오.
    2교시 2절(메타데이터 설계)과 11주차(필터 검색·출처 표기)의 근거가 됩니다.

  Loader 별 '단위' 가 다릅니다 ★

      Text  →  1개 (파일 전체)          자동 metadata: source
      PDF   →  페이지 수만큼 ★          자동 metadata: source, page ★
      CSV   →  행 수만큼 ★              자동 metadata: source, row
      Web   →  1개                      자동 metadata: source, title

  📌 핵심 메시지: **"쓰레기를 넣으면 쓰레기가 나옵니다."**
     파이프라인 6단계 중 ①에서 망가진 것은 ⑥에서 절대 복구되지 않습니다.

  ⚠️ 임포트 오류가 나면: Loader 클래스의 위치(langchain_community.document_loaders)는
     버전에 따라 옮겨진 적이 있습니다. requirements.txt 대로 설치했는지 먼저 확인하세요.

실행:
    python loaders.py            # 로컬 파일 3종 (Web 은 건너뜀 · 네트워크 없음)
    python loaders.py --web      # Web Loader 까지 (⚠️ 30명 동시 호출 주의)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

# WebBaseLoader 는 User-Agent 가 없으면 경고를 냅니다. 미리 채워 둡니다.
import os

os.environ.setdefault("USER_AGENT", "dongyang-ai-class/2026 (education)")

from langchain_community.document_loaders import CSVLoader, TextLoader, WebBaseLoader

DATA = Path(__file__).parent / "data"

# ★ data/학칙.pdf 는 실제 학칙 PDF(28쪽)입니다. 저장소에는 없으니 학교 규정집에서 받아 넣으세요.
#    https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do
#    ⚠️ '표가 깨지는 페이지' 와 '머리말이 섞이는 페이지' 는 진짜 PDF 라야 보입니다.
#       합성 PDF 로는 이 절의 교육 효과가 나오지 않습니다.
PDF_PATH = DATA / "학칙.pdf"

# 💡 학교 공지 등 실제로 접근 가능한 주소로 바꿔 보세요.
WEB_URL = "https://example.com/notice"


def show(name: str, docs) -> None:
    print("=" * 60)
    if not docs:
        print(f"[{name}]  건너뜀")
        return
    print(f"[{name}]  Document 개수: {len(docs)}")
    print("metadata:", docs[0].metadata)  # ★ 무엇이 '자동으로' 붙는가
    print("본문 앞 200자:")
    print(docs[0].page_content[:200].strip())
    print()


def load_text():
    """① 텍스트 — 파일 전체가 Document 1개. 구조 정보가 전혀 없다."""
    return TextLoader(str(DATA / "규정.txt"), encoding="utf-8").load()


def load_pdf():
    """② PDF — 페이지 단위로 쪼개진다. metadata 에 page 가 붙는다 ★"""
    if not PDF_PATH.exists():
        print("=" * 60)
        print(f"""[PDF]  건너뜀 — {PDF_PATH} 가 없습니다. ⚠️ 아래 주소에서 받아 data/학칙.pdf 로 넣으세요.
         https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do
         (README «사전 준비» 의 다운로드 명령 한 줄이면 됩니다)

  ⚠️ 이 절의 핵심(파싱 난점)은 **진짜 PDF** 라야 보입니다.
     data/학칙.pdf 는 실제 학칙(28쪽)입니다. 1쪽 머리말과 표를 확인하세요.

     ┌─────────────────────────────────┐
     │  동양미래대학교 학칙        - 7 -│  ← 머리말·페이지 번호가 본문에 섞인다
     ├─────────────────────────────────┤
     │  제25조(휴학)                    │
     │  ┌──────┬──────┬──────┐         │
     │  │ 구분  │ 기간  │ 비고  │        │  ← 표가 한 줄 텍스트로 뭉개진다 ⚠️
     │  └──────┴──────┴──────┘         │
     │  ① 질병, 창업, 출산 ...          │  ← 단이 나뉘면 순서가 뒤섞인다
     └─────────────────────────────────┘
""")
        return []

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("[PDF]  건너뜀 — pip install pypdf 가 필요합니다. ⚠️")
        return []

    return PyPDFLoader(str(PDF_PATH)).load()


def load_csv():
    """③ CSV — 행 단위로 Document 가 만들어진다. 열 이름이 본문에 섞인다 ★"""
    return CSVLoader(str(DATA / "faq.csv"), encoding="utf-8").load()


def load_web():
    """④ 웹 — 메뉴·광고·푸터가 본문에 섞인다 ⚠️

    ⚠️ 30명이 같은 주소를 동시에 때리면 차단될 수 있습니다. 조별 순차 실행. ★
    """
    try:
        return WebBaseLoader(WEB_URL).load()
    except Exception as e:
        print(f"[Web]  건너뜀 — {type(e).__name__}: {str(e)[:60]} ⚠️")
        return []


def main() -> None:
    show("Text", load_text())
    show("PDF", load_pdf())
    show("CSV", load_csv())

    if "--web" in sys.argv:
        show("Web", load_web())
    else:
        print("=" * 60)
        print("[Web]  건너뜀 — 켜려면: python loaders.py --web  (⚠️ 조별 순차 실행)\n")

    print("=" * 60)


if __name__ == "__main__":
    main()
