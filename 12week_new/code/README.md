# 12주차 실습 코드 — LangGraph (1) 상태 그래프 · 분기 · 순환

**2026년 11월 20일 (금)** / 최신인공지능 12주차

> **"체인은 되돌아갈 수 없습니다."**
> 5~11주차에 만든 것은 모두 `prompt | model | parser` 처럼 왼쪽에서 오른쪽으로 한 번 흐르고 끝났습니다.
> 오늘은 **갈라지고(분기) 되돌아가는(순환)** 흐름을 LangGraph 로 만듭니다.

> 📌 **12주차 실습은 이 폴더(`12week_new/code`)를 씁니다.** `rag_router_graph.py` 는 10주차 `10week_new/code/index_recursive` 인덱스를 이어 씁니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| **실습 1-①** | 1교시 판서 · 3교시 실습 | [`first_graph.py`](first_graph.py) | 첫 그래프 — State 정의 → 노드 2개 → `compile()` → `invoke()` |
| **실습 1-②** ★★ | 1교시 판서 · 3교시 실습 | [`reducer_demo.py`](reducer_demo.py) | **리듀서** 있을 때 / 없을 때 — 덮어쓰기 vs 누적 (LLM 없음 · 즉시) |
| **실습 2** ★★ | 2교시 판서 · 3교시 실습 | [`branch_graph.py`](branch_graph.py) | **조건부 엣지** — 질문 유형 분류 → 세 갈래 / `--plain` 으로 분류 프롬프트 비교 |
| 판서 · 실습 | 2교시 판서 · 3교시 실습 | [`recursion_demo.py`](recursion_demo.py) | 종료 조건이 없으면? — `recursion_limit` 과 `GraphRecursionError` (LLM 없음 · 즉시) |
| **실습 3** ★★ | 2교시 판서 · 3교시 실습 | [`reflection.py`](reflection.py) | **Reflection 루프** — 생성 → 평가 → 재작성 / `--judge llm` 으로 자기 평가와 비교 |
| **실습 4** | 3교시 실습 | [`visualize.py`](visualize.py) | 그래프 그림 (ASCII · Mermaid) + LangSmith 실행 경로 |
| **프로젝트 연결** ★ | 3교시 | [`rag_router_graph.py`](rag_router_graph.py) | **학칙 질문만 RAG 로** — 슬랙 봇 처리를 분기 그래프로 (과제 5 의 골격) |
| **실습 5 · 미니 프로젝트** ★ | 3교시 | [`my_bot_graph.py`](my_bot_graph.py) | **내 봇의 갈래로** — ✏️ 고칠 곳 네 군데(갈래 표 · 실패 시 갈래 · 노드 · 연결), `--route-only` 로 분류 확인 |
| 공용 | — | [`faiss_io.py`](faiss_io.py) | 인덱스 읽기 — 경로에 한글이 있어도 되게 한다 (10·11주차와 같은 파일) |
| 견본 | — | [`.env.example`](.env.example) | 반복 상한 · recursion_limit · 인덱스 위치 |

## 사전 준비

```bash
# 10·11주차까지 쓰던 가상환경을 그대로 사용합니다
venv\Scripts\Activate.ps1          # Windows PowerShell
source venv/Scripts/activate       # Git Bash

pip install -r requirements.txt     # ★ langgraph · grandalf 가 새로 추가

python reducer_demo.py              # ★ LLM 없이 즉시 끝납니다 — messages 2개 · count 2 가 보이면 설치 정상
```

- `ollama list` 에 `gemma3:4b` 와 `qwen3-embedding:0.6b` 가 있어야 합니다 (10주차와 같음).
- `rag_router_graph.py` 는 10주차 `index_recursive/` 가 필요합니다. 없으면 `10week_new/code` 에서 10주차 실습 3을 먼저 하세요.
- 실습 4 에서 LangSmith 실행 경로를 보려면 루트 `.env` 의 `LANGSMITH_TRACING=true` 와 키가 필요합니다 (6주차 설정 그대로).

## 실행 순서

```bash
# ── 1교시 (판서) 에서 읽는 코드 → 3교시 실습 1 ────────────────────────
python first_graph.py                # 노드 2개 직렬 — 최종 상태에 키 3개가 모두 남는다
python reducer_demo.py               # ★★ 리듀서 없으면 A 의 말이 사라진다

# ── 2교시 (판서) 에서 읽는 코드 → 3교시 실습 2 · 3 ────────────────────
python branch_graph.py               # 질문 3개가 세 갈래로
python branch_graph.py --plain       # ★ 분류 프롬프트를 한 줄로 줄이면 경로가 바뀌는가?
python recursion_demo.py             # 종료 조건이 없으면 — GraphRecursionError
python reflection.py                 # ★★ 규칙 검사로 평가 — 몇 회차에 통과하나?
python reflection.py --judge llm     # ★ 같은 모델이 스스로 평가하면?

# ── 3교시 실습 4 · 프로젝트 연결 ─────────────────────────────────────
python visualize.py                  # reflection 그래프 — Mermaid 에서 되돌아가는 화살표 찾기
python visualize.py branch           # 분기 그래프
python visualize.py router           # 학칙 / 일반 분기 그래프
python rag_router_graph.py           # ★ 학칙 질문만 검색으로 간다
python rag_router_graph.py --route-only "졸업하려면 학점 몇 점 필요해?"   # 분류만 (빠름)

# ── 3교시 실습 5 · 미니 프로젝트 ──────────────────────────────────────
python my_bot_graph.py --route-only  # ★ 고치기 전 그대로 — 분류만 (인덱스 없이도 됨)
python my_bot_graph.py --route-only "내 질문"   # ROUTES 를 내 갈래로 바꾼 뒤
python visualize.py my_bot_graph     # 내 그래프 그림 — 과제 5 ② · README 에 붙일 Mermaid
```

> ⏱ `reflection.py` 는 반복마다 LLM 을 부르므로 1~4분 걸립니다. 실습실 전체가 동시에 돌리면 더 걸립니다.
> 반복 상한은 `.env` 의 `WEEK12_MAX_ATTEMPTS`(기본 3), 안전망은 `WEEK12_RECURSION_LIMIT`(기본 10) 입니다.

> ⚠️ **`recursion_limit` 을 꼭 직접 적으세요.** 아무것도 안 적으면 LangGraph 기본 한도는 **10007 단계** 입니다
> (인터넷의 예전 자료에는 25 로 나옵니다). LLM 노드가 있는 그래프라면 사실상 무한 루프입니다.

## 실습 기록표 — 직접 채우세요 ★

**실습 2 — 분류 프롬프트에 따라 경로가 바뀌는가**

| 질문 | 기본 (설명 + 예시) | `--plain` (한 줄) | 맞는 경로 |
|---|---|---|---|
| 357 곱하기 4891은? | | | math |
| 우리 학교 휴학 규정 알려줘 | | | rag |
| 오늘 기분이 좀 별로야 | | | chat |

**실습 3 — Reflection 회차별 기록**

| 회차 | 글자 수 | 평가(지적 사항) | LLM 호출 누계 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

| | `rule` (규칙 검사) | `--judge llm` (자기 평가) |
|---|---|---|
| 몇 회차에 끝났나 | | |
| 마지막 줄 «규칙 검사로 다시 보면» | | |

## 5주차 Self-Consistency 와 오늘의 Reflection

| | Self-Consistency (5주) | Reflection (오늘) |
|---|---|---|
| 구조 | **병렬** — 여러 개를 동시에 | **순환** — 하나를 반복해서 |
| 방식 | 여러 답을 뽑아 투표 | 쓰고 → 평가 → 고쳐 쓰기 |
| 구현 | `batch()` | 그래프의 되돌아가는 엣지 |
| 비용 | N배 (한 번에) | 반복 횟수만큼 (점점) |

> **기법은 실행 구조를 요구합니다.** 다수결은 병렬이 없으면 못 하고, Reflection 은 순환이 없으면 못 합니다.

## 과제 5 — 조건 분기를 포함한 LangGraph 그래프

| 제출물 | 내용 |
|---|---|
| ① 코드 | State 스키마(`TypedDict`, 필요한 곳에 리듀서) · 노드 3개 이상(분류 노드 + 경로별 노드) · 조건부 엣지로 2개 이상 경로 |
| ② 그래프 그림 | `visualize.py` 의 ASCII 캡처 · Mermaid 텍스트 · PNG 중 하나 |
| ③ 실행 결과 | **서로 다른 경로를 타는 입력 2개 이상** |
| ④ LangSmith 캡처 | 실행 경로(어느 노드를 거쳤는지)가 보이게 |

채점에서 보는 것: 조건에 따라 실제로 다른 노드로 가는가 · 상태가 노드 사이에 올바르게 전달되는가(리듀서 선택) · 순환이 있다면 종료 조건이 있는가.

> 💡 **미니 프로젝트와 같은 그래프로 만드세요.** `rag_router_graph.py` 를 출발점으로 삼아 분류 기준과 경로를 내 프로젝트에 맞게 바꾸면,
> 그대로 14주차 최종 제출물의 골격이 됩니다. 마감·제출 방법은 수업 공지를 따르세요.

## 실습 5 — 내 봇의 갈래로 바꾸기 (`my_bot_graph.py`)

1. 고치기 전에 `python my_bot_graph.py --route-only` 로 한 번 돌립니다 (예시 4문항이 학칙 · 도움말 · 일반 · 학칙으로 갑니다).
2. 파일에서 **«✏️»** 를 찾아 ① `ROUTES`(갈래 이름 · 기준 · 예시 질문)부터 내 프로젝트에 맞게 바꿉니다. 분류 프롬프트는 이 표로 자동으로 만들어집니다.
3. ② `FALLBACK` · `RouteName`, ④ `FIRST_NODE` 의 갈래 이름을 ① 과 **똑같이** 맞춥니다. 갈래를 늘렸다면 ③ 노드 함수와 `add_node` 도 추가합니다.
4. 내 질문 5개 이상으로 `--route-only` — 틀린 질문은 ① 의 **예시**에 넣고 다시 돌립니다.
5. `python visualize.py my_bot_graph` 의 Mermaid 를 프로젝트 README 에 붙입니다.

> 학칙 갈래는 `rag_router_graph.py` 의 `retrieve` · `rag_answer` 를 가져다 쓰므로 `rag_router_graph.py` · `faiss_io.py` 와 같은 폴더에 두세요.
> 슬랙 봇에 붙이는 법은 아래와 같습니다 — `rag_router_graph` 대신 `my_bot_graph` 에서 `answer` 를 가져오면 됩니다.

## 미니 프로젝트로 이어 가기 — 슬랙 봇에 분기 그래프 붙이기

10주차 슬랙 봇(`10week_new/code/slack_bot_mode/slack_rag_bot.py`)은 질문마다 `ask(backend, question, meta)` 를 불러 **언제나 검색**했습니다.
`rag_router_graph.py` 의 `answer()` 는 질문 하나를 받아 슬랙에 올릴 글 하나를 돌려주므로, 봇에서 답을 만드는 자리만 바꾸면 됩니다.

```python
# slack_rag_bot.py 의 «RAG — ① 자리 잡기 → ② 찾아서 답하기 → ③ 자리를 답으로» 부분 (예시)
from rag_router_graph import answer as graph_answer   # rag_router_graph.py · faiss_io.py 를 봇 폴더로 복사해 옵니다

holder = post_holder(client, channel, thread, "⏳ 생각 중…")
text = graph_answer(question)                          # 학칙 질문 → 검색 + 근거 조항 / 그 외 → 바로 답
client.chat_update(channel=channel, ts=holder["ts"], text=text)   # 슬랙 스레드의 자리를 답으로 (10주차와 같음)
```

> 봇 파일에는 이미 `answer(event, client)` 라는 함수가 있어서 이름이 겹칩니다. 위처럼 `as graph_answer` 로 이름을 바꿔 가져오세요.

- 학칙.pdf 인덱스(쪽 번호까지 나오는 것)를 쓰려면 `.env` 에 `WEEK12_INDEX=...` 를 적습니다. 이 경로는 **`rag_router_graph.py` 파일이 있는 폴더 기준**입니다.
  - 파일을 12week_new/code 에 둔 채로 쓸 때: `WEEK12_INDEX=../../10week_new/code/slack_bot_mode/index_pdf`
  - 파일을 봇 폴더(`10week_new/code/slack_bot_mode/`)로 복사했을 때: `WEEK12_INDEX=index_pdf`
- 처리 가운데를 MCP 로 고른 팀은 `retrieve` 노드 자리에 MCP 도구 호출을 넣으면 됩니다 — 그래프 모양은 같습니다.
- ⚠️ 일반 경로는 근거 없이 답하므로 모델이 지어낼 수 있습니다. 실행해 보고 일반 경로의 답이 사실인지 확인하세요 (13주차 Agentic RAG 에서 이어 갑니다).
