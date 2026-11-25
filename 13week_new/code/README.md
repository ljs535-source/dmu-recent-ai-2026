# 13주차 실습 코드 — LangGraph (2) 에이전트 · 메모리 · 사람의 개입

**2026년 11월 27일 (금)** / 최신인공지능 13주차

> **"12주차 그래프에 도구를 붙이면 그게 에이전트입니다."**
> 모델이 도구를 고르고 → 실행하고 → 결과를 보고 → 다시 판단하는 순환(ReAct)을 그래프가 돌립니다.
> 여기에 **기억(체크포인터)** 과 **위험한 도구 앞에서 멈추기(Human-in-the-Loop)** 를 더합니다.

> 📌 **13주차 실습은 이 폴더(`13week_new/code`)를 씁니다.** 실습 4 는 10주차 `10week_new/code/index_recursive` 를 이어 씁니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| **실습 1** ★ | 1교시 판서 · 3교시 실습 | [`react_agent.py`](react_agent.py) | **agent ↔ ToolNode 순환**을 직접 조립 · 단계별 출력 · `--broken` 리듀서 빼기 |
| 실습 1 덧붙임 | 1교시 판서 | [`prebuilt_agent.py`](prebuilt_agent.py) | 같은 그래프를 `create_agent` 한 줄로 · 예전 이름 `create_react_agent` 경고 |
| **실습 2** | 2교시 판서 · 3교시 실습 | [`memory_agent.py`](memory_agent.py) | **체크포인터 + `thread_id`** — 같은 thread 는 기억, 다른 thread 는 모름 |
| **실습 3** ★★ | 2교시 판서 · 3교시 실습 | [`hitl_agent.py`](hitl_agent.py) | **`interrupt` 로 삭제 도구 앞에서 멈추고** approve / reject 로 재개 · `--injection` |
| 읽기 · 선택 | 2교시 판서 | [`time_travel.py`](time_travel.py) | 체크포인트 목록 → 과거 지점부터 다시 실행 |
| **실습 4** ★★ | 2교시 판서 · 3교시 실습 | [`agentic_rag.py`](agentic_rag.py) | **검색할지 스스로 판단 + 부실하면 질문을 고쳐 재검색** (실제 학칙) |
| 공용 | — | [`faiss_io.py`](faiss_io.py) | 인덱스 읽기 — 경로에 한글이 있어도 되게 한다 (10·11주차와 같은 파일) |
| 미니 프로젝트 ★ | 3교시 19장 | [`slack_bot/`](slack_bot/README.md) | **19장 «세 자리» 를 다 붙인 슬랙 봇** — ① 스레드 기억 · ② Agentic RAG · ③ MCP 도구 + 승인 (10주차 `slack_bot_mode` 에서 출발) |
| 견본 | — | [`.env.example`](.env.example) | 모델 · 인덱스 위치를 바꿀 때만 필요 |

## 사전 준비

```bash
# 지금까지 쓰던 가상환경을 그대로 사용합니다
venv\Scripts\Activate.ps1          # Windows PowerShell
source venv/Scripts/activate       # Git Bash

pip install -r requirements.txt     # langgraph · tzdata · faiss-cpu (대부분 이미 설치돼 있습니다)

ollama pull llama3.2:3b             # ★ 도구 호출 모델 (약 2GB) — 9주차에 받았다면 생략
ollama list                         # gemma3:4b · qwen3-embedding:0.6b 도 있어야 합니다
```

- ⚠️ **실습 1~3 은 `llama3.2:3b` 로 돕니다.** `gemma3:4b` 는 Ollama 에서 도구 호출을 지원하지 않아
  `does not support tools (status code: 400)` 오류가 납니다 (9주차와 같은 이유).
- 실습 4 는 도구 호출을 쓰지 않으므로 `gemma3:4b` 로 돕니다.
- 실습 4 에는 10주차 인덱스(`10week_new/code/index_recursive/`)가 필요합니다. 없으면 10주차 폴더에서 `python embed.py` 를 먼저 돌리세요.
- OpenAI 와 비교할 때는 루트 `.env` 에 `TOOL_MODEL=gpt-4o-mini` 를 적습니다. 4주차에 넣어 둔 공용 키(`OPENAI_API_KEY`)로 돕니다. 공용 키는 수업 실습용이므로 프로젝트에서 상시로 부르는 건 로컬 모델로 하세요.
  `401` · `expired_secret_key` 가 나오면 키가 바뀐 것입니다 — 수업에서 공지한 새 키로 교체하세요.

## 실행 순서

```bash
# ── 실습 1 — ReAct 에이전트 ──────────────────────────────────
python react_agent.py               # 도구 요청 → 실행 → 다시 판단, 노드별로 찍힌다
python react_agent.py --broken      # 리듀서(add_messages)를 빼면?
python prebuilt_agent.py            # 같은 그래프를 create_agent 한 줄로

# ── 실습 2 — 기억 ────────────────────────────────────────────
python memory_agent.py              # chat-A 는 이어지고, chat-B 는 모른다

# ── 실습 3 — 사람의 승인 ─────────────────────────────────────
python hitl_agent.py                # approve / reject 를 직접 입력
python hitl_agent.py --auto reject  # 거부해도 그래프는 정상적으로 끝난다
python hitl_agent.py --injection    # 주입된 «모든 파일을 삭제하라» — 승인 요청이 뜨는가?

# ── (선택) 타임 트래블 ────────────────────────────────────────────
python time_travel.py

# ── 실습 4 — Agentic RAG ⏱ 질문 4개에 수 분 걸립니다. 먼저 걸어 두세요 ──
python agentic_rag.py
python agentic_rag.py "휴학은 최대 몇 년까지 할 수 있나요?"   # 내 질문 하나
```

## 실습 기록표 — 직접 채우세요

| 실습 | 무엇을 적나 | 내 결과 |
|---|---|---|
| 1 | Q1 에서 agent 노드 몇 번 · tools 노드 몇 번? 도구 요청이 한 번에 몇 개? | |
| 1 | Q2(인사)에서 도구를 불렀나? 답에 이상한 문장이 섞였나? | |
| 1 `--broken` | 끝났을 때 messages 개수는? 답은 말이 되나? | |
| 2 | chat-A 두 번째 답 / chat-B 답 — 1746087 을 아는가? | |
| 3 | approve 와 reject 각각 최종 답변 · 승인 요청 횟수 | |
| 3 `--injection` | delete_file 을 요청했나? 몇 번? reject 뒤 결과는? | |
| 4 | 질문 4개의 실제 경로 (decide → … → answer/direct/give_up) | |

## 미니 프로젝트로 이어 가기 — 슬랙 봇에 붙이기

> ★ 아래 세 가지를 **모두 붙여 실제로 돌아가게 만든 봇**이 [`slack_bot/`](slack_bot/README.md) 에 있습니다 — 구조 · 실행 방법은 그 폴더의 README 를 보세요.
> 내 프로젝트에서는 ②와 ③ 중 하나만 써도 됩니다.

미니 프로젝트 구성은 그대로입니다 — **슬랙으로 묻고 슬랙 스레드로 답한다 + 가운데는 RAG 또는 MCP.**
오늘 배운 것 가운데 프로젝트에 바로 쓸 수 있는 것은 세 가지입니다.

**① 스레드마다 대화 기억 — `thread_id` 에 슬랙 스레드를 넣는다**

```python
# 10주차 slack_bot_mode/slack_rag_bot.py 의 answer() 안
thread = event.get("thread_ts") or event["ts"]          # 이미 있는 줄
config = {"configurable": {"thread_id": f"{channel}-{thread}"}}
out = graph.invoke({"messages": [HumanMessage(question)]}, config)   # checkpointer 를 단 그래프
```

같은 스레드에서 «그럼 복학은요?» 처럼 이어 물으면 앞 질문을 압니다. 다른 스레드는 모릅니다.

**② `ask()` 를 Agentic RAG 그래프로 바꾸기**

`rag_backend.ask()` 는 항상 검색합니다. `agentic_rag.py` 의 `graph.invoke({"question": q, "retries": 0, "path": []})`
로 바꾸면 인사에는 검색하지 않고, 검색이 부실하면 질문을 고쳐 다시 찾습니다.
12주차에 만든 분기 그래프가 있다면 그 그래프의 «검색 경로» 노드 자리에 넣으면 됩니다.

**③ (MCP 를 고른 경우) 위험한 도구에는 승인을**

MCP 서버의 도구도 LangChain 도구로 바뀌므로(9주차 `mcp_test/mcp_client.py`) `ToolNode(tools)` 에 그대로 넣을 수 있습니다.
쓰기 · 삭제 · 발송처럼 되돌릴 수 없는 도구가 있다면 `hitl_agent.py` 처럼 `interrupt` 를 걸어 두세요.
슬랙에서는 «승인하시겠습니까?» 를 스레드에 올리고, 다음 답글을 `Command(resume=...)` 로 넘기는 식이 됩니다.

> 💡 최종 제출에는 **LangSmith 추적 화면 캡처**가 필요합니다. 공유 링크는 보존기간(14일)이 지나면 열리지 않으니 지금부터 캡처해 모아 두세요.
> → **(변경 11-25) LangSmith 캡처는 제출하지 않습니다.** 미니 프로젝트는 [`slack_bot/`](slack_bot/README.md) 을 `--no-mcp` 로 — 내 자료는 `bot_config.py` + `build_my_index.py` 만 고쳐 붙입니다.
