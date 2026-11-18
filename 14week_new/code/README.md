# 14주차 실습 코드 — LangGraph (3) 멀티 에이전트 · 스트리밍 · 서비스화

**2026년 12월 4일 (금)** / 최신인공지능 14주차 — 학기 마지막 진도

> **멀티 에이전트는 정확도를 호출 수로 삽니다.** 나누는 것은 역할이 정말 갈릴 때입니다.
> **"배포하면 끝" 이 아니라 "배포하면 시작" 입니다.**

> 📌 **14주차 실습은 이 폴더(`14week_new/code`)를 씁니다.** 10주차 `10week_new/code/index_recursive`(실제 학칙 인덱스)를 이어 씁니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| **실습 1** ★★ | 1교시 판서 · 3교시 실습 | [`supervisor.py`](supervisor.py) | **Supervisor 패턴** — 관리자 + 담당자 3 (학칙 검색 · 곱셈 · 작성), 담당자는 **서브그래프** / `--single` 로 단일 에이전트와 LLM 호출 수 비교 |
| **실습 2** ★ | 2교시 판서 · 3교시 실습 | [`streaming.py`](streaming.py) | **상태 단위 / 토큰 단위 스트리밍** · `--subgraphs` 로 담당자 안쪽까지 |
| 실습 2 (이어서) | 2교시 · 3교시 | [`slack_progress.py`](slack_progress.py) | 상태 단위 진행 표시를 **슬랙 메시지 하나(chat_update)** 로 — 슬랙 접속 없이 가짜 client 로 확인 |
| **실습 3** ★★ | 2교시 판서 · 3교시 실습 | [`langgraph.json`](langgraph.json) · [`studio_hitl.py`](studio_hitl.py) | **`langgraph dev` 로컬 서버 + Studio** — supervisor 그래프 보기 · 13주차 승인(HITL)을 화면에서 |
| **실습 4** | 2교시 판서 · 3교시 실습 | [`ops_feedback.py`](ops_feedback.py) | **운영 관점 LangSmith** — 개념 / `--demo` 피드백 붙이기 / `--stats` 지연 P50·P95 |
| 공용 | — | [`faiss_io.py`](faiss_io.py) | 인덱스 읽기 — 경로에 한글이 있어도 되게 한다 (10·11주차와 같은 파일) |
| 견본 | — | [`.env.example`](.env.example) | 모델 · 안전장치 · 인덱스 위치를 바꿀 때만 필요 |

## 사전 준비

```bash
# 지금까지 쓰던 가상환경을 그대로 사용합니다
venv\Scripts\Activate.ps1          # Windows PowerShell
source venv/Scripts/activate       # Git Bash

pip install -r requirements.txt     # ★ langgraph-cli[inmem] 이 새로 추가 (실습 3)

ollama list                         # llama3.2:3b · gemma3:4b · qwen3-embedding:0.6b 가 있어야 합니다
ollama pull llama3.2:3b             # 없으면 (약 2.0GB)
```

- **모델이 둘인 이유** — `bind_tools` 로 도구를 부르는 곳(calculator · `--single` · studio_hitl)은 `llama3.2:3b` 를 씁니다.
  ⚠️ `gemma3:4b` 는 Ollama 에서 **도구 호출을 지원하지 않습니다**(9주차 실측). 도구를 부르지 않는 supervisor · researcher 요약 · writer 는 한국어가 나은 `gemma3:4b`.
- ⚠️ `llama3.2:3b` 는 시스템 메시지와 함께 **한국어 문자열 인자**(검색어 등)를 넘기면 도구 호출이 깨지기도 합니다
  (도구 호출 대신 `{"name": "search_rules", "parameters": {"query": "\uce3…"}}` 같은 글자를 답으로 냄).
  그래서 researcher 는 모델이 도구를 고르게 하지 않고 **코드가 검색한 뒤 모델이 요약**합니다. 내 프로젝트에서 같은 증상이 보이면 이 방식을 쓰거나,
  `.env` 에 `TOOL_MODEL=kimjk/llama3.2-korean`(한국어 학습판, 2025년 수업 모델)을 넣어 비교해 보세요.
- 10주차 인덱스(`10week_new/code/index_recursive/`)가 없으면 검색 도구가 «[오류] 인덱스가 없습니다» 를 돌려줍니다.
  11주차 `python rag_common.py` 를 한 번 돌리면 다시 만들어집니다.
- ⚠️ `langgraph-cli[inmem]` 을 설치하면 `protobuf` · `grpcio` 버전이 바뀔 수 있습니다. 11주차 Chroma 실습이 그 뒤에 안 되면
  14주차용 가상환경을 따로 만들어 `pip install -r requirements.txt` 하세요.
- 실행하면 `langchain-community is being sunset …` 경고가 나올 수 있습니다. **오류가 아닙니다** (11주차와 같음).

## 실행 순서

```bash
# ── 1교시 (판서) 에서 읽는 코드 → 3교시 실습 1 ───────────────────────
python supervisor.py            # 멀티 에이전트 — 담당자 순서와 LLM 호출 수를 적어 두세요
python supervisor.py --single   # 같은 질문을 단일 에이전트(도구 2개)로 — 호출 수 비교

# ── 2교시 (판서) 에서 읽는 코드 → 3교시 실습 2 · 3 · 4 ─────────────────
python streaming.py --updates   # 상태 단위 — 노드가 끝날 때마다 한 줄
python streaming.py --subgraphs # 담당자(서브그래프) 안쪽 노드까지
python streaming.py --tokens    # 토큰 단위 — writer 의 답만 글자로
python slack_progress.py        # 슬랙 메시지 하나가 어떻게 고쳐지는지 (접속 없음)

langgraph dev                   # 실습 3 — 이 폴더에서 실행, 출력의 Studio 주소를 브라우저로

python ops_feedback.py          # 개념 정리 (LLM 호출 없음)
python ops_feedback.py --demo   # run 하나에 👍 피드백 붙이기 (LangSmith 키 필요)
python ops_feedback.py --stats  # 내 프로젝트 최근 run 의 오류율 · 지연 P50/P95
```

> ⚠️ `supervisor.py` 는 LLM 을 여러 번 부릅니다. 실습실에서 다 같이 돌리면 느려집니다 — 옆 사람과 번갈아 돌리세요.

## 실습 1 — 관찰할 것

| 관찰 | 생각할 점 |
|---|---|
| supervisor 가 **여러 번** 판단한다 | 배분 → 실행 → 보고를 읽고 → 다시 판단 (12주차 순환) |
| 담당자마다 맡은 일이 하나 | researcher 는 검색, calculator 는 도구 `multiply` 하나 — 9주차 원칙(도구 2~3개)을 각자 지킨다 |
| 담당자 안쪽 구조가 다르다 | researcher = 검색 → 요약 (11주차 RAG 모양) / calculator = 모델 ↔ 도구 (13주차 ReAct 모양) — 둘 다 서브그래프 |
| 담당자는 **보고 한 줄**만 남긴다 | 서브그래프 안에서 오간 도구 호출은 관리자에게 넘기지 않는다 |
| LLM 호출 수 (멀티 vs `--single`) | 나누면 정확해질 수 있지만 호출이 는다 — 7주차 판단 틀(정확도 · 토큰 · 지연) |
| 담당자를 엉뚱하게 고를 때 | 소형 모델의 한계 → `MAX_STEPS` 에 닿으면 writer 가 정리하고 끝낸다 |

**기록표** (직접 채우기)

| | 담당자 순서 | LLM 호출 수 | 걸린 시간 | 답이 맞았나 (4주 × 7 = 28일) |
|---|---|---|---|---|
| `supervisor.py` | | | | |
| `supervisor.py --single` | | | | |

## 실습 3 — `langgraph dev` 와 Studio

```bash
cd 14week_new/code
langgraph dev
```

`langgraph.json` 의 핵심은 `graphs` 입니다 — `"이름": "파일경로:변수명"`.

```json
{
  "dependencies": ["."],
  "graphs": {
    "supervisor": "./supervisor.py:graph",
    "hitl": "./studio_hitl.py:graph"
  },
  "env": "../../.env"
}
```

실행하면 이런 줄이 나옵니다.

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Studio UI 주소를 브라우저로 엽니다(LangSmith 로그인 화면이 나오면 6주차 계정으로). 그래프는 **내 PC 의 서버**에서 돌고, 브라우저 화면만 smith.langchain.com 에서 옵니다.

- `"env"` 는 저장소 맨 위의 `.env`(하나뿐)를 가리킵니다. 주차 폴더에 `.env` 를 새로 만들지 마세요.
- 실행하면 이 폴더에 `.langgraph_api/` 폴더가 생깁니다(서버가 메모리 내용을 잠깐 저장하는 곳). 지워도 되고, 커밋하지 마세요.
- `studio_hitl.py` 의 `graph` 는 **체크포인터 없이** compile 합니다. 서버가 저장소를 붙여 주기 때문입니다.
- `langgraph dev` 가 안 될 때는 같은 승인 그래프를 터미널에서 확인할 수 있습니다: `python studio_hitl.py --auto reject` (자동 거부) · `python studio_hitl.py --auto approve` (자동 승인)

**Studio 에서 확인할 것**

- [ ] `supervisor` 그래프 그림에 **되돌아가는 화살표**(researcher · calculator → supervisor)가 보이는가
- [ ] 질문을 넣고 실행하면 **지금 도는 노드**가 표시되는가
- [ ] `hitl` 그래프에 «report.pdf 를 삭제해줘» → `tools` 에서 **멈추고** 승인 요청이 보이는가
- [ ] resume 값으로 `approve` / `reject` 를 각각 넣어 결과가 달라지는가 (거부도 정상 흐름)

> ⚠️ `langgraph dev` 는 **개발용**입니다. 저장은 메모리(끄면 사라짐) · 인증 없음 — 서비스로 열어 두지 마세요.

## 안 될 때

| 증상 | 원인 | 조치 |
|---|---|---|
| `does not support tools` | 도구 담당 모델이 gemma3:4b | `.env` 에 `TOOL_MODEL=llama3.2:3b` (기본값) |
| `[오류] 10주차 인덱스가 없습니다` | 인덱스 경로 | 11주차 `python rag_common.py` 로 다시 만들기 · `WEEK14_INDEX` |
| `langgraph: command not found` | CLI 미설치 / venv 미활성 | `pip install "langgraph-cli[inmem]"` · venv 활성화 확인 |
| `SystemError: ConsoleRenderer with colors=True on Windows requires the colorama package` | Windows 에 colorama 없음 | `pip install colorama` (requirements.txt 에 있음) |
| `GraphLoadError … includes a custom checkpointer` | `compile(checkpointer=…)` 한 그래프를 등록함 | 서버가 저장소를 붙이므로 `compile()` 만 — `studio_hitl.py` 처럼 |
| `Graph import exceeded the expected startup time` (warning) | 불러올 때 모델 · 모듈 준비가 느림 | 경고일 뿐입니다. `Application started up` 이 나오면 정상 |
| `langgraph dev` 포트 오류 | 2024 포트 사용 중 | `langgraph dev --port 2025` |
| Studio 가 안 열림 | 브라우저가 로컬 주소 접속을 막음 | 다른 브라우저(Chrome · Edge) / 출력의 주소를 직접 복사 |
| Studio 의 hitl 이 안 멈춤 | 모델이 delete_file 을 안 부름 | 질문을 «report.pdf 파일을 삭제해줘» 처럼 분명히 |
| 매우 느림 | 멀티 에이전트는 호출이 많다 | 정상입니다 — 그게 이 패턴의 대가입니다 |
| `--demo` 에서 키 안내만 나옴 | `LANGSMITH_API_KEY` 없음 | 루트 `.env` 에 키 · `LANGSMITH_TRACING=true` |
| 한글이 `UnicodeEncodeError` | 콘솔 코드페이지 | `chcp 65001` 또는 `set PYTHONUTF8=1` |

## 미니 프로젝트로 이어 가기

- **진행 표시** — `slack_progress.py` 의 `answer_with_progress(event, client)` 가 10주차 `slack_rag_bot.py` 의 `answer()` 와 같은 모양입니다.
  내 그래프를 `graph` 자리에 넣으면 «⏳ 학칙을 찾는 중…» → «✅ 답» 으로 메시지 하나가 고쳐집니다.
  ⚠️ `chat_update` 는 노드 단위(수 초에 한 번)로만 — 토큰마다 고치면 슬랙 rate limit 에 걸립니다.
- **피드백** — 답할 때 `run_id = uuid.uuid4()` 를 정해 `invoke(..., {"run_id": run_id})` 로 넘기고, 답 메시지에 달린 👍/👎 반응을 `create_feedback(run_id, …)` 으로 붙일 수 있습니다.
- **최종 제출** — 코드 · `README.md`(10주차 템플릿 6항목) · **LangSmith 추적 캡처(필수, 무료 보존 14일)**. 학칙.pdf 는 공개 저장소에 올리지 마세요.

## 꼭 이해해야 할 지점

- 멀티 에이전트가 필요한 이유 · Supervisor 의 동작 (배분 → 실행 → 보고 → 다시 판단)
- 서브그래프 — compile() 결과가 노드가 된다, 담당자는 결과만 보고한다
- 종료 조건 세 겹 — writer → END · MAX_STEPS · recursion_limit
- 스트리밍 2종과 `stream_mode` (values · updates · messages)
- `langgraph dev` 와 Studio 로 할 수 있는 것, 개발 서버와 운영 배포의 차이
- 운영 LangSmith — 모니터링(평균이 아니라 P95) · 피드백(run_id) · 지속 평가(회귀)
