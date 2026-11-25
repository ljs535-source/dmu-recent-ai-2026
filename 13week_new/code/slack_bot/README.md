# slack_bot — 3교시 19장 «오늘 것을 붙이는 세 자리» 가 실제로 도는 슬랙 봇

10주차 [`slack_bot_mode`](../../../10week_new/code/slack_bot_mode/README.md)(슬랙으로 묻고 학칙 PDF 로 찾아 답하는 봇)에서 출발해,
13주차에 배운 세 가지를 **모두** 붙인 미니 프로젝트 예시입니다.

| 19장의 자리 | 이 봇에서 | 배운 곳 |
|---|---|---|
| **① 스레드마다 기억** | `thread_id = f"{channel}-{thread_ts}"` + 체크포인터 — 같은 스레드의 «그럼 휴학은요?» 를 앞 질문으로 풀어 답한다 | 2교시 §1 · `memory_agent.py` |
| **분류** (골격) | 12주차 `my_bot_graph.py` 의 ROUTES 표 — 학칙 / 도구 / 도움말 / 일반 | 12주차 3교시 |
| **② Agentic RAG** | search → grade → (rewrite → search)* → answer \| give_up | 2교시 §4 · `agentic_rag.py` |
| **③ MCP 도구 + 승인** | MCP 서버의 메모 도구를 `ToolNode` 에 · 쓰기 · 삭제 도구는 `interrupt` 로 멈추고 슬랙에서 승인 | 1교시 · 2교시 §2 · `hitl_agent.py` · 9주차 MCP |

> 📌 19장 그림 아래 줄 그대로 — **«가운데는 RAG 또는 MCP 하나 이상. ②와 ③을 다 할 필요는 없습니다.»**
> 이 봇은 예시라서 둘 다 넣었습니다. 내 프로젝트에서는 하나만 남겨도 됩니다.

> 📌 **(변경 11-25) 미니 프로젝트의 기준 코드는 이 폴더입니다.** 요건은 **② RAG** 이고, **③ MCP 도구를 1개 이상 추가하는 것을 권장합니다**(넣지 않으면 `--no-mcp` 로 실행 · 승인은 선택). LangSmith 캡처는 요건에서 빠졌습니다.

> ★ **미니 프로젝트 — 내 데이터 봇**: 지금까지 모은 내 RAG 데이터(PDF · MD · TXT · CSV)를 이 봇에 연결합니다.
> 고칠 파일은 [`bot_config.py`](bot_config.py) **하나**, 돌릴 명령은 [`build_my_index.py`](build_my_index.py) **하나**입니다 → 아래 [«✏️ 내 데이터 봇 만들기»](#️-내-데이터-봇-만들기--미니-프로젝트).

## 구조

```
   슬랙  @봇 질문 (스레드)
     │   WebSocket (xapp) — Socket Mode
     ▼
   slack_agent_bot.py            ← 입구 · 출구 (10주차 slack_rag_bot.py 에서 두뇌만 바꿈)
     │  ① thread_id = "채널-스레드ts"
     │  ⏸ 이 스레드에 승인 대기 중이면 → 이번 메시지는 «승인 / 거부» 답으로 처리
     ▼
   bot_graph.Brain.ask() / resume()
     │
     ▼  ┌──────────────────── 내 봇의 그래프 (bot_graph.py · 체크포인터 InMemorySaver) ────────────────────┐
        │                                                                                              │
        │  START → contextualize → classify ─┬─ 학칙  → search → grade ─┬─ answer ──────────────┐       │
        │          (후속 질문 풀기)  (12주차)  │                          ├─ rewrite → search (순환) │       │
        │                                    │                          └─ give_up ──────────────┤       │
        │                                    ├─ 도구  → tool_agent → tools ──▶ tool_answer ────────┤       │
        │                                    │          (llama3.2:3b)  ⏸ interrupt   (결과만 알림)   │       │
        │                                    │                         (add_note · delete_note) │       │
        │                                    ├─ 도움말 → help (정해진 글 · LLM 없음) ─────────────┤       │
        │                                    └─ 일반  → general ──────────────────────────────────┴→ END │
        └──────────────────────────────────────────────────────────────────────────────────────────────┘
     │                                       │
     │  ② 검색: index_pdf/ (학칙.pdf 전체)     │  ③ 도구: mcp_server.py (MCP 서버 · 자식 프로세스)
     ▼                                       ▼
   슬랙 스레드  ← 답 + 📚 근거 조항 + 경로   /   ⏸ «승인이 필요합니다» → (몇 분 뒤) @봇 승인 → 이어서 실행
```

### 한 질문이 지나가는 길

| 질문 (같은 스레드) | 경로 | 무엇을 보나 |
|---|---|---|
| `몇 주를 결석하면 제적되나요?` | contextualize → classify(학칙) → search → grade → answer | ② 근거 조항 «제29조(제적)» |
| `그럼 휴학은 최대 몇 년까지 할 수 있어요?` | contextualize(**앞 대화로 풀기**) → classify → search → … | ① 같은 thread_id 라 앞 질문을 안다 |
| `메모 남겨줘: 금요일 3시 휴학 상담` | … → classify(도구) → tool_agent → tools **⏸** | ③ 쓰기 도구 → 승인 요청 |
| `@봇 승인` (같은 스레드) | tools(재개) → tool_answer | `Command(resume="approve")` — tools 노드가 **처음부터** 다시 돈다 |
| `내 메모 보여줘` | … → tool_agent → tools → tool_answer | 읽기 도구라 승인 없이 바로 |

### 모델 두 개

| 역할 | 모델 | 이유 |
|---|---|---|
| contextualize · classify · grade · rewrite · answer · general | `MODEL` = **gemma3:4b** | 판단은 구조화 출력(JSON) — `agentic_rag.py` 와 같음 |
| tool_agent (도구 고르기) | `TOOL_MODEL` = **llama3.2:3b** | gemma3:4b 는 도구 호출 미지원 (`does not support tools (400)`) |
| tool_answer (도구 결과 알리기) | `MODEL` = gemma3:4b | 도구가 없는 모델이라 **더 부를 수 없다** — 아래 «소형 모델 안전장치» |

### 소형 모델 안전장치 — 실측에서 나온 것

| 장치 | 왜 넣었나 (이 PC 실측) | 바꾸는 곳 |
|---|---|---|
| **contextualize 는 «그럼 · 그거 · 거기 · 아까…» 가 있을 때만** | 모든 질문을 다시 쓰게 했더니 «그럼 휴학은 최대 몇 년?» 이 «…결석하면 제적되나요?» 로, «메모 남겨줘: 금요일 3시 휴학 상담» 이 «…예약해 주시겠습니까?» 로 **주제가 바뀌었다**. 다시 쓴 질문에 원래 낱말이 하나도 없으면 원문을 쓴다 | `FOLLOW_UP` |
| **도구 갈래는 슬랙 원문 그대로** | 다시 쓴 질문을 넘겼더니 «내 메모 보여줘» 에 `add_note` 를 불렀다 | `tool_agent` |
| **도구 왕복 1번** (`MAX_TOOL_ROUNDS`) | 메모를 저장한 뒤 답하지 않고 `list_notes` → **`delete_note`(아무도 안 시킴)** → `list_notes` ×13 → `recursion_limit`. 시키지 않은 삭제는 ⏸ 승인이 막았다 — **HITL 이 실제로 일한 장면** | `.env` `WEEK13_MAX_TOOL_ROUNDS` |
| `recursion_limit` 에 걸려도 봇은 살아 있게 | 위 루프에서 오류로 봇이 죽었다 | `Brain._run` |

## 파일

| 파일 | 역할 | 출처 |
|---|---|---|
| [`slack_agent_bot.py`](slack_agent_bot.py) | **슬랙 봇** — 입구 · 출구 · ① thread_id · 승인 답 처리 | 10주차 `slack_rag_bot.py` 를 고침 |
| [`bot_graph.py`](bot_graph.py) | **내 봇의 그래프** + `Brain`(봇이 부르는 손잡이) · 터미널 데모 | 새로 (12주차 분류 골격 + 13주차 실습 1~4) |
| [`mcp_server.py`](mcp_server.py) | ③ MCP 서버 — `list_notes` · `add_note` · `delete_note` (메모는 `notes.json`) | 새로 (9주차 FastMCP 모양) |
| [`bot_config.py`](bot_config.py) | ✏️ **내 데이터 봇 설정** — 주제 · 분류 예시 · 문서 이름 · 인덱스 · 데모 질문 · 출처 표시 | 새로 (기본값 = 학칙 봇) |
| [`build_my_index.py`](build_my_index.py) | ✏️ **내 데이터 → 인덱스** `my_data/` → `index_my/` (PDF · MD · TXT · CSV) | 새로 (10주차 적재 · 분할 · 메타 · 임베딩 · 저장) |
| `my_data_example/` | 형식 견본 — 가상의 사진 동아리 `회칙.md` · `faq.csv` | 새로 |
| [`rag_backend.py`](rag_backend.py) | 10주차 RAG — «항상 검색» 기준점 (봇은 쓰지 않음) | 10주차 그대로 (경로 주석만) |
| [`build_index.py`](build_index.py) | 학칙.pdf → `index_pdf/` (PDF 를 바꿀 때만) | 10주차 (경로만 — 10주차 `data/` · `split_compare.py` 를 읽음) |
| `index_pdf/` | 학칙.pdf 전체 인덱스 (119 청크 · qwen3-embedding:0.6b) | 10주차에서 복사 |
| [`requirements.txt`](requirements.txt) | 13주차 패키지 + `slack_bolt` · `langchain[mcp]` · `pypdf` | |

> ★ `agentic_rag.py`(실습 4)는 10주차 `index_recursive`(학칙 **제5·7장** 정리본)를 씁니다.
> 이 봇은 10주차 슬랙 봇과 같은 **학칙.pdf 전체** 인덱스 `index_pdf/` 를 씁니다 — 졸업 · 등록금 · 학사경고도 답할 수 있습니다.

## 준비

| 확인 | 어떻게 | 안 되면 |
|---|---|---|
| ① 패키지 | `pip install -r requirements.txt` (공용 venv 면 이미 있음) | |
| ② 모델 | `ollama list` 에 `gemma3:4b` · `llama3.2:3b` · `qwen3-embedding:0.6b` | `ollama pull …` |
| ③ 슬랙 토큰 | 9주차 `socket_hello.py` 에서 pong — 루트 `.env` 의 `SLACK_BOT_TOKEN`(xoxb-) · `SLACK_APP_TOKEN`(xapp-) | 9주차 실습 1 |
| ④ 슬랙 앱 이벤트 | `app_mentions:read` · `chat:write` 권한, `app_mention` 이벤트 구독 (10주차와 같은 앱) | 9주차 3교시 |

루트 `.env` (저장소 맨 위 한 개 — 9 · 10주차와 같은 파일):

```
SLACK_BOT_TOKEN=xoxb-…
SLACK_APP_TOKEN=xapp-…
SLACK_ALLOWED_USERS=U0…          # (권장) 남의 요청도 대신 승인할 수 있는 사람 — 비우면 «요청한 사람» 만
MODEL=gemma3:4b                  # (선택) 판단 · 답 모델
TOOL_MODEL=llama3.2:3b           # (선택) 도구 모델 — 도구 호출을 지원해야 합니다
EMBED_MODEL=qwen3-embedding:0.6b # ⚠️ index_pdf/ 를 만든 모델과 같아야 합니다
```

## 실행 순서 (이 폴더에서)

```bash
# 1단계 — 슬랙 없이 그래프만 (★ 봇이 이상할 때 범인을 가르는 기준점)
python bot_graph.py                    # 데모 대화 5개 — 승인 질문이 나오면 approve / reject 입력
python bot_graph.py --auto approve     # 승인 자동 (reject 도 해 보기)
python bot_graph.py --chat             # 직접 입력 (같은 스레드로 이어짐) — 빈 줄이면 끝
python bot_graph.py --no-mcp           # MCP 없이 — RAG 만 (MCP 는 권장 — 6단계)

# (참고) 10주차 «항상 검색» RAG 와 견주기
python rag_backend.py 검색 "몇 주를 결석하면 제적되나요?"
python rag_backend.py "몇 주를 결석하면 제적되나요?"

# 2단계 — 슬랙 봇 켜기 (끝낼 때는 Ctrl+C)
python slack_agent_bot.py
python slack_agent_bot.py --no-mcp
```

> ⚠️ **10주차 봇은 끄고 켜십시오.** 같은 앱 토큰(xapp)으로 두 봇이 붙어 있으면 이벤트가 둘 중 한쪽으로 갈립니다.
> ⚠️ 체크포인터가 `InMemorySaver` 라서 **봇을 껐다 켜면 스레드 기억과 승인 대기가 모두 사라집니다** (2교시 8장 «단기 vs 장기»).
> 💡 시작할 때 `Starting MCP server 'week13-notes'` 로그가 보이면 MCP 서버가 자식 프로세스로 뜬 것입니다(봇이 끝나면 함께 꺼집니다).

## 슬랙에서 쓰는 법

| 메시지 | 하는 일 |
|---|---|
| `@봇 몇 주를 결석하면 제적되나요?` | ② 학칙에서 찾아 답 + 📚 근거 조항 · 쪽 + 경로 |
| (같은 스레드) `@봇 그럼 휴학은요?` | ① 앞 질문을 기억해 «휴학» 질문으로 풀어 답 |
| `@봇 메모 남겨줘: 금요일 3시 휴학 상담` | ③ ⏸ «승인이 필요합니다» — 같은 스레드에 `@봇 승인` / `@봇 거부` |
| `@봇 내 메모 보여줘` · `@봇 2번 메모 지워줘` | ③ 보기는 바로, 지우기는 승인 후 |
| `@봇 검색 졸업 학점` | (10주차 그대로) 어떤 조항이 뽑혔는지만 — LLM 없이 즉시 |
| `@봇 도움말` · `@봇` | 사용법 |

**승인 규칙** — 승인 대기 중인 스레드에서는 다음 멘션이 «답» 으로 처리됩니다.
`승인` · `approve` → 실행 / `거부` · `reject` · `취소` → 실행 안 함 / 그 밖의 말 → «먼저 답해 주세요» 안내.
**요청한 사람**(또는 `SLACK_ALLOWED_USERS`)만 승인할 수 있습니다 — 다른 사람이 `@봇 승인` 하면 🔒 로 막습니다.

### 승인이 «기다리지 않는» 이유 — 2교시 14장 ⭐ 줄

`interrupt` 는 그래프 실행을 **끝내 버립니다**(`invoke` 가 돌아온다). 봇은 «승인이 필요합니다» 를 올리고 함수를 끝냅니다.
몇 분 뒤 `@봇 승인` 이 오면 **그 이벤트**가 `Brain.resume()` = `Command(resume="approve")` 로 그래프를 다시 부릅니다.
터미널 실습(`hitl_agent.py`)의 바깥 `for` 루프(①)가 슬랙에서는 «다음 멘션 이벤트» 로 바뀐 것입니다. 그 사이 봇은 다른 사람의 질문을 받습니다.

## 실행 결과 (2026-11-24, 이 PC — GPU 없음 · 공용 venv langgraph 1.2.12 · gemma3:4b · llama3.2:3b)

**터미널** `python bot_graph.py --auto approve`

| 스레드 | 질문 | 경로 | 결과 | 시간 |
|---|---|---|---|---|
| A | 몇 주를 결석하면 제적되나요? | contextualize → classify(학칙) → search → grade → answer | «연속하여 4주를 초과하여 결석 … 제29조(제적)» 📚 제29조 p.7 | 98초 |
| A | 그럼 휴학은 최대 몇 년까지 할 수 있어요? | (같음) — search 1위 **제25조(휴학)** | 휴학 조항으로 답 · 스레드 메시지 2 → **4개** (① 기억) | 130초 |
| B | 안녕하세요! | contextualize → classify(일반) → general | 검색 없이 인사 | 28초 |
| B | 메모 남겨줘: 금요일 3시 휴학 상담 | … → tool_agent **⏸** → (approve) → tools → tool_answer | `add_note({'text': '금요일 3시 휴학 상담'})` → «메모를 저장했습니다» | 66초 + 13초 |
| B | 내 메모 보여줘 | … → tool_agent → tools → tool_answer | `list_notes` — 승인 없이 바로 | 18초 |
| B | 1번 메모 지워줘 | … → tool_agent **⏸** → (approve) → tools → tool_answer | `delete_note({'note_id': 1})` → «삭제되었습니다» | 13초 + 4초 |

**슬랙 흐름** — `slack_agent_bot.answer()` 를 가짜 슬랙 client 로(실제 슬랙에는 아무것도 올리지 않음)

| 메시지 (같은 스레드) | 봇의 반응 |
|---|---|
| `메모 남겨줘: 금요일 3시 휴학 상담` | «⏳ 생각 중…» → «⏸ *승인이 필요합니다* — `add_note(...)`» 로 고침 |
| `안녕?` (승인 대기 중) | «⏸ 이 스레드에 승인 대기 중인 요청이 있습니다 …» 안내만 |
| 다른 사람의 `승인` | «🔒 승인은 @요청한사람 님 또는 허용된 사람만 …» |
| 요청한 사람의 `승인` | 재개 → «금요일 3시 휴학 상담 메모를 저장했습니다.» |
| `1번 메모 지워줘` → `거부` | ⏸ → 재개 → «삭제 요청이 거부되었습니다» — `notes.json` 에 1번 메모가 **그대로 남음** |
| `검색 졸업 학점` · `도움말` | 10주차와 같은 검색 결과(1위 제42조(졸업)) · 사용법 |

**실제 슬랙 접속** — `python slack_agent_bot.py` → 인덱스 확인 ✅ · MCP 도구 3개 · `Bolt app is running!` (접속만 확인, 실제 멘션 답은 위의 가짜 client 로 확인)

> 📌 이 PC 는 GPU 가 없어 학칙 답 하나에 30~130초입니다. 그래서 봇은 먼저 «⏳» 를 달고, 답이 오면 그 메시지를 고칩니다(9 · 10주차와 같음).
> 📌 모델이 근거 밖 내용을 섞을 때가 있습니다 — 그래서 📚 근거와 경로를 함께 보여 줍니다.

## ✏️ 내 데이터 봇 만들기 — 미니 프로젝트

지금까지 모은 미니 프로젝트용 RAG 데이터를 이 봇에 붙입니다. **봇 코드(`bot_graph.py` · `slack_agent_bot.py`)는 고치지 않습니다.**

```
   my_data/  ──▶ build_my_index.py ──▶ index_my/ ──┐
   (PDF·MD·TXT·CSV)   (10주차 ①~⑤ 그대로)            │
                                                   ▼
   bot_config.py  ✏️ 주제 · 예시 질문 · 문서 이름 · INDEX  ──▶  bot_graph.py  ──▶  slack_agent_bot.py
```

### 1단계 — 내 문서를 `my_data/` 에

| 형식 | 이렇게 읽힌다 | 출처 표시 (슬랙 «📚 근거») | 이렇게 준비하면 좋다 |
|---|---|---|---|
| **PDF** | 쪽마다 → 500자로 분할 | `파일.pdf p.3` | 스캔 PDF(글자가 그림)는 안 읽힌다 — 글자를 드래그로 고를 수 있는지 먼저 확인 |
| **MD** | **`## 제목` 마다** → 500자로 분할 | `## 제목 · 파일.md` | 주제마다 `## 제목` 을 붙인다 — 제목이 곧 출처가 된다 (10주차 학칙의 «제N조» 와 같은 생각) |
| **TXT** | 파일 하나 → 500자로 분할 | `파일.txt` | 문단 사이에 빈 줄 |
| **CSV** | **한 행 = 한 청크** («열이름: 값» 줄들) | `첫 칸 값 · 파일.csv 행N` | 첫 줄은 열 이름, **첫 열은 질문 · 이름처럼 제목이 될 값** (FAQ · 일정표 · 목록) |

```bash
mkdir my_data              # 또는 my_data_example 을 복사해서 형식을 보고 바꾼다
# 내 파일을 넣는다 — 한글 파일 이름 OK. 저장 인코딩은 UTF-8 (엑셀 CSV 는 cp949 도 읽음)
```

> ⚠️ **10 · 12주차에 이미 인덱스를 만들었다면** 2단계를 건너뛰고 `bot_config.py` 의 `INDEX` 에 그 폴더 경로만 적어도 됩니다.
> 단, 그 인덱스를 만든 **임베딩 모델과 `.env` 의 `EMBED_MODEL` 이 같아야** 합니다(다르면 차원이 달라 검색이 깨집니다).
> 출처 표시는 그 인덱스의 metadata 에 따라 달라집니다(`article` · `title` · `source` · `page` · `row` 를 읽습니다).

### 2단계 — 인덱스 만들기

```bash
python build_my_index.py --show     # ★ 먼저 — 저장 없이 청크 수 · metadata 만 (임베딩 없음 · 즉시)
python build_my_index.py            # my_data/ → index_my/
python build_my_index.py 다른폴더 --out index_club   # 폴더 · 인덱스 이름을 바꿀 때
```

`--show` 에서 볼 것 — metadata 의 `title` · `page` · `row` 가 **슬랙에 출처로 보여도 괜찮은 값인가.**
청크가 너무 길면(가장 긴 것이 수천 자) 문서에 빈 줄 · `## 제목` 을 더 넣는다.

### 3단계 — `bot_config.py` 고치기 (✏️ 표시 값만)

| 값 | 무엇 | 예 (파일 맨 아래 «동아리 봇» 예시) |
|---|---|---|
| `INDEX` | 읽을 인덱스 폴더 | `"index_my"` — 또는 루트 `.env` 에 `BOT_INDEX=index_my` |
| `TOPIC` · `TOPIC_DESC` · `TOPIC_EXAMPLES` | **분류 갈래** 이름 · 기준 · 예시 질문 2~3개 (12주차 ROUTES 표) | `"동아리"` · «회칙 · 일정 · 회비 …를 찾아봐야 답할 수 있는 질문» · `["회비 얼마야?", …]` |
| `DOC_NAME` | 프롬프트 속 «어디에서 찾나» · «…에서 찾지 못했습니다» | `"동아리 자료"` |
| `REWRITE_HINT` · `REWRITE_EXAMPLE` | ② 재작성 때 쓰게 할 낱말 · 예시 한 줄 | `"회칙 용어(회비·정기모임·장비대여 등)"` |
| `CITE_EXAMPLE` | 답 끝에 밝힐 출처의 예 | `"제1조 (회비) · 회칙.md"` |
| `BOT_INTRO` · `TOPIC_HELP` | 도움말 · 사용법 문구 | `"동아리 안내 봇"` |
| `DEMO` | 터미널 데모 질문 — **같은 스레드에서 «그럼 · 그거» 로 이어 묻는 줄**을 꼭 | `("thread-A", "그럼 그거 언제까지 내요?")` |
| `USE_RAG` · `USE_MCP` | RAG 만 / MCP 만 하는 팀 | RAG 만이면 `USE_MCP = False` |

> ⚠️ 글자 안에 중괄호 `{ }` 를 쓰지 마세요 — 프롬프트 틀이 변수로 읽습니다.
> 💡 분류가 자꾸 «일반» 으로 새면 `TOPIC_EXAMPLES` 에 **구어체 · 줄임말 질문**을 더 넣으세요 (12주차 «헷갈리는 예시일수록 좋다»).

### 4단계 — 터미널 → 5단계 슬랙

```bash
python bot_graph.py --no-mcp        # RAG 만 먼저 — 경로 · 📚 근거 · 같은 스레드 기억 확인
python bot_graph.py                 # MCP 까지
python slack_agent_bot.py           # 슬랙에서
```

### (권장) 6단계 — MCP 추가하기 (추가 11-25)

**MCP 도구를 1개 이상 추가하는 것을 권장합니다.** 다만 **`--no-mcp` 로 RAG 봇이 도는 것을 먼저 확인한 뒤에** 얹으세요 — MCP 가 막혀도 RAG 봇은 그대로 남습니다.

1. `ollama pull llama3.2:3b` — 도구를 고르는 모델 (gemma3:4b 는 도구 호출 미지원 → 400 오류)
2. `bot_config.py` 에서 `USE_MCP = True`
3. `--no-mcp` 를 **빼고** 실행: `python bot_graph.py` → `python slack_agent_bot.py` (로그에 `Starting MCP server` 가 보이면 켜진 것)
4. 먼저 들어 있는 메모 도구로 확인: «내 메모 보여줘» (바로 실행) · «메모 남겨줘: 금요일 3시 상담» (⏸ 승인 요청)
5. 내 도구를 추가하려면(1개 이상) 네 곳을 함께 고칩니다 — 메모 도구는 지워도 되고 남겨도 됩니다

| 고칠 곳 | 할 일 |
|---|---|
| `mcp_server.py` | `@mcp.tool` 함수 — docstring 이 도구 설명, 타입 힌트가 인자 (9주차). `print()` 금지 |
| `bot_graph.py` `ROUTES["도구"]` | 분류 설명과 예시 질문을 내 도구에 맞게 |
| `bot_graph.py` `TOOL_SYSTEM` | «이런 말이면 이 도구» 기준을 새 도구 이름으로 |
| `bot_graph.py` `DANGEROUS` | 승인이 필요한 도구 이름 — 쓰기 · 삭제 · 발송만 |

> 💡 소형 모델은 도구를 여러 번 부르는 경향이 있습니다(실측). 도구는 2~3개, 이름과 docstring 은 분명하게.

### 견본 실측 — `my_data_example/` 를 동아리 봇으로 (2026-11-24, 이 PC)

`build_my_index.py my_data_example` → Document 8개(MD 5절 — 머리말 + 제1~4조 · CSV 3행) → 청크 8개.
`bot_config` 를 파일 맨 아래 «동아리 봇» 예시 값으로 바꾸고 `--no-mcp` 로:

| 스레드 | 질문 | 검색 1위 | 답 |
|---|---|---|---|
| A | 회비는 얼마예요? | 제1조 (회비) · 회칙.md | «학기마다 20,000원» ✅ |
| A | **그럼 그거** 언제까지 내야 해요? | ① «회비는 언제까지 납부해야 하나요?» 로 풀림 → 제1조 | «3월 31일(1학기) · 9월 30일(2학기)» ✅ |
| B | 카메라 빌리면 며칠 쓸 수 있어? | 제3조 (장비 대여) · 회칙.md | «최대 3일» ✅ |
| B | 전시회는 언제 해? | 전시회는 언제 하나요? · faq.csv 행2 | «11월 셋째 주 · 학생회관 로비» ✅ |
| C | 안녕! | (classify → 일반, 검색 없음) | 인사 |
| C | 도서관 열람실은 몇 시까지 열어요? (자료에 없음) | 엉뚱한 3개 | «동아리 자료에서 찾지 못했습니다» ✅ — 단, grade 는 True 로 잘못 판정, answer 가 걸렀다 |

> 📌 gemma3:4b 는 맞게 답한 뒤 **근거에 있던 다른 내용(출사 일정 등)을 덧붙이거나** «찾지 못했습니다» 를 붙일 때가 있습니다.
> 그래서 답 아래에 📚 근거와 경로를 함께 보여 줍니다. 프로젝트 문서에 «이런 오답이 있었다» 를 적는 것도 평가 재료입니다(7 · 11주차).

### 그 밖에 바꿀 수 있는 것

| 바꿀 것 | 어디서 |
|---|---|
| 내 MCP 도구 | `mcp_server.py` 의 `@mcp.tool` 함수 — docstring 이 도구 설명, 타입 힌트가 인자 (9주차). 도구를 바꾸면 `bot_graph.py` 의 `ROUTES["도구"]` 예시 · `TOOL_SYSTEM` 기준도 함께 |
| 승인이 필요한 도구 | `bot_graph.py` 의 `DANGEROUS` — **쓰기 · 삭제 · 발송**만 (모두 걸면 승인 피로 · 2교시 18장) |
| 재작성 · 도구 왕복 · 단계 상한 | `.env` 의 `WEEK13_MAX_RETRY`(2) · `WEEK13_MAX_TOOL_ROUNDS`(1) · `WEEK13_RECURSION_LIMIT`(25) |
| 학칙 인덱스를 다시 만들 때 | `python build_index.py` (10주차 — 학칙.pdf 의 «제N조» 분할) |
| 봇을 껐다 켜도 기억 | `InMemorySaver` → `SqliteSaver` (`pip install langgraph-checkpoint-sqlite`) — 2교시 8장 |

## LangSmith 에서 볼 것

- 슬랙 질문 1건 = trace 1건 (`slack-agent`), 태그 `week13-slack-bot`, metadata 에 `slack_user` · `slack_channel`
- trace 안에 **노드 이름**(contextualize · classify · search · grade · …)이 차례로 보입니다 — 슬랙 답 아래 «경로» 와 같아야 합니다.
- 승인으로 멈춘 질문은 trace 가 **두 건**입니다(멈출 때까지 한 건 · 재개 후 한 건). 같은 `thread_id` 로 묶어 보세요.
- 💡 최종 제출에는 LangSmith 추적 화면 **캡처**가 필요합니다 — 공유 링크는 14일 뒤 열리지 않습니다.
  → **(변경 11-25) LangSmith 캡처는 제출하지 않습니다.** 미니 프로젝트 제출물은 `build_my_index.py --show` · `bot_graph.py --no-mcp` · 슬랙 대화 캡처와 README 입니다(13주차 체크리스트 «최종 제출 미리 보기»).
