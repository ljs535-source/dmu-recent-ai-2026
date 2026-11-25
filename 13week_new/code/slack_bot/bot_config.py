"""✏️ 내 데이터 봇 설정 — 미니 프로젝트에서는 **이 파일 하나만** 고칩니다

   이 봇(bot_graph.py · slack_agent_bot.py)은 «무엇을 찾아 답하는 봇인가» 를 전부 여기서 읽습니다.
   기본값은 학칙 봇(10주차 학칙.pdf 전체)입니다. 내 데이터 봇으로 바꾸는 순서:

     ① 내 문서를  my_data/  폴더에 넣는다             (PDF · MD · TXT · CSV — 10주차 2교시 «형식»)
     ② python build_my_index.py                      → index_my/ 가 생긴다 (임베딩 · 저장)
     ③ 아래 «✏️» 표시 값을 내 주제로 바꾼다            (INDEX · TOPIC · 예시 질문 · 문서 이름 …)
     ④ python bot_graph.py                           → 터미널에서 먼저 확인
     ⑤ python slack_agent_bot.py                     → 슬랙에서

   ★ 10 · 12주차에 이미 만든 인덱스가 있으면 ① ② 를 건너뛰고 INDEX 에 그 폴더 경로만 적어도 됩니다.
     단, 그 인덱스를 만든 임베딩 모델과 .env 의 EMBED_MODEL 이 같아야 합니다 (다르면 차원이 달라 검색이 깨집니다).

  ⚠️ 아래 글자 안에 중괄호 { } 를 쓰지 마세요 — 프롬프트 틀(ChatPromptTemplate)이 변수로 읽습니다.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── ✏️ ① 인덱스 — 어느 폴더의 벡터 인덱스를 읽나 (이 폴더 기준 경로) ─────────────
#    학칙 봇: "index_pdf"  (10주차 slack_bot_mode 에서 복사해 온 것)
#    내 데이터: "index_my"  (build_my_index.py 가 만든 것)  ·  12주차 인덱스: "../../../12week_new/code/…"
INDEX = os.getenv("BOT_INDEX", "index_pdf")
INDEX_DIR = (HERE / INDEX).resolve()

# ── ✏️ ② 무엇을 하는 봇인가 ──────────────────────────────────────────────
USE_RAG = True   # 문서에서 찾아 답하는 갈래 (② Agentic RAG)  — MCP 만 하는 팀은 False
USE_MCP = True   # 메모 도구 갈래 (③ MCP 도구 + 승인)          — RAG 만 하는 팀은 False (또는 --no-mcp)

BOT_INTRO = "학칙 안내 + 메모 봇"                 # 도움말 · 사용법 첫 줄
TOPIC = "학칙"                                    # 분류 갈래 이름 (짧게 — «도구» «도움말» «일반» 과 겹치면 안 됨)
TOPIC_DESC = "우리 학교 학칙(휴학·복학·자퇴·제적·성적·졸업·등록금 등 학사 규정)을 찾아봐야 답할 수 있는 질문"
TOPIC_EXAMPLES = [                                # 분류 예시 2~3개 — 헷갈리는 구어체를 넣을수록 좋다 (12주차)
    "휴학은 몇 년까지 돼?",
    "몇 주 결석하면 제적이야?",
    "학교 그만두려면 어떻게 해?",
]
DOC_NAME = "대학 학칙"                            # 프롬프트 속 «어디에서 찾나» · «…에서 찾지 못했습니다»
REWRITE_HINT = "규정 용어(휴학·복학·제적·자퇴·성적·졸업 등)"   # 재작성할 때 쓰게 할 낱말
REWRITE_EXAMPLE = "'학교 그만두고 싶어요' → '자퇴 절차는 어떻게 되나요?'"
CITE_EXAMPLE = "제29조"                            # 답 끝에 밝힐 출처의 예 (조 번호 · 파일 이름 · 쪽 · 행 …)
TOPIC_HELP = "휴학·복학·제적·성적·졸업을 물으면 학칙 PDF 에서 찾아 근거 조항과 함께 답합니다"

# 터미널 데모 질문 — (스레드, 질문). 같은 스레드에서 이어 묻는 줄을 꼭 넣으세요 (① 기억 확인)
DEMO = [
    ("thread-A", "몇 주를 결석하면 제적되나요?"),
    ("thread-A", "그럼 휴학은 최대 몇 년까지 할 수 있어요?"),
    ("thread-B", "안녕하세요!"),
    ("thread-B", "메모 남겨줘: 금요일 3시 휴학 상담"),
    ("thread-B", "내 메모 보여줘"),
    ("thread-B", "1번 메모 지워줘"),
]


# ── 출처 한 줄 — 슬랙 답 아래 «📚 근거» 에 쓰인다 (10주차 metadata 의 회수) ───────────
def source_label(doc) -> str:
    """metadata 로 «어디서 왔나» 한 줄을 만든다.

    학칙 인덱스(10주차 build_index.py): article «제29조(제적)» · page
    내 데이터 인덱스(build_my_index.py): title(문서 제목 · CSV 첫 칸) · page(PDF 쪽) · row(CSV 행)
    """
    m = doc.metadata
    head = m.get("article") or m.get("title") or m.get("chapter") or ""
    where = m.get("source", "?")
    if "page" in m:
        where += f" p.{m['page']}"
    if "row" in m:
        where += f" 행{m['row']}"
    return f"{head} · {where}" if head else where


# ─────────────────────────────────────────────────────────────────────────
# ✏️ 예시 — 내 데이터 봇으로 바꾼 모습 (주석을 풀고 위 값 대신 쓰세요)
#
# INDEX = os.getenv("BOT_INDEX", "index_my")
# INDEX_DIR = (HERE / INDEX).resolve()
# BOT_INTRO = "동아리 안내 봇"
# TOPIC = "동아리"
# TOPIC_DESC = "우리 동아리의 회칙 · 모임 일정 · 회비 · 장비 대여 규칙을 찾아봐야 답할 수 있는 질문"
# TOPIC_EXAMPLES = ["회비 얼마야?", "카메라 빌리려면 어떻게 해?", "정기 모임 언제야?"]
# DOC_NAME = "동아리 자료"
# REWRITE_HINT = "회칙 용어(회비·정기모임·장비대여·가입·탈퇴 등)"
# REWRITE_EXAMPLE = "'돈 언제 내?' → '회비 납부 기한은 언제인가요?'"
# CITE_EXAMPLE = "회칙.md"
# TOPIC_HELP = "회칙 · 일정 · 회비 · 장비 대여를 물으면 동아리 자료에서 찾아 출처와 함께 답합니다"
# DEMO = [("thread-A", "회비는 얼마예요?"), ("thread-A", "그럼 그거 언제까지 내요?"), ("thread-B", "안녕!")]
