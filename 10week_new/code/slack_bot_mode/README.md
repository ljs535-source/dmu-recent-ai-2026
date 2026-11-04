# slack_bot_mode — 슬랙에서 묻고, 학칙 PDF 에서 찾아서 답한다

9주차 슬랙 봇(`slack_llm_bot.py`)에 **10주차 RAG** 를 붙인 코드입니다.
근거 문서는 **`../data/학칙.pdf`** (실제 학칙 28쪽, 규정번호 2-0-1, 2024.07.01. 개정) 입니다.
⚠️ 이 PDF 는 저장소에 없습니다 — [학교 규정집 다운로드 링크](https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do)에서 받아 넣으세요 (아래 «준비» ③).
미니 프로젝트의 기본 모양 — **슬랙으로 묻고, 가운데서 RAG 로 찾고, 슬랙으로 답한다** — 을 그대로 보여 줍니다.

```
   ../data/학칙.pdf ──▶ build_index.py ──▶ index_pdf/          (한 번만 · 1교시 적재 + 2교시 분할 + 3교시 임베딩)

   슬랙  @봇 몇 주를 결석하면 제적되나요?
     │
     ▼
   slack_rag_bot.py   ← 9주차 봇 (입구 · 출구)
     │
     ▼  rag_backend.ask()
   ① 검색   index_pdf/ 에서 가까운 청크 3개
   ② 생성   근거 3개 + 질문 → gemma3:4b (또는 gpt-4o-mini)
     │
     ▼
   슬랙 스레드   답 + 📚 근거  제29조(제적) · 학칙.pdf p.7
```

## 파일

| 순서 | 파일 | 역할 | 9주차에서 대응하는 파일 |
|---|---|---|---|
| ① | [`build_index.py`](build_index.py) | **학칙.pdf → 봇 전용 인덱스** `index_pdf/` | (10주차 `embed.py`) |
| ② | [`rag_backend.py`](rag_backend.py) | **RAG 쪽만** 터미널에서 확인 — 검색 · 생성 · 비교 | `llm_backends.py` |
| ③ | [`slack_rag_bot.py`](slack_rag_bot.py) | 슬랙 봇 — `ask()` 만 RAG 로 바뀌었다 | `slack_llm_bot.py` |
| | [`requirements.txt`](requirements.txt) | 10주차 패키지 + `slack_bolt` · `langchain-openai` | |

> ★ **9주차 봇과 바뀐 곳은 거의 import 한 줄입니다.** `llm_backends.ask` → `rag_backend.ask`
> 봇(입구 · 출구)과 두뇌(RAG)가 나뉘어 있어서 **두뇌만 갈아 끼웁니다.** MCP 를 붙일 때도 같은 자리입니다.

> ⚠️ 3교시 `embed.py` 의 `../index_recursive/` (학칙.md 정리본) 는 **건드리지 않습니다.** 11·13주차가 그대로 이어 씁니다.
> 봇은 PDF 전체로 만든 **`index_pdf/`** 를 따로 씁니다.

## build_index.py 가 하는 일 — 10주차 세 교시가 한 파일에

| 단계 | 하는 일 | 배운 곳 |
|---|---|---|
| ① 적재 | `PyPDFLoader` — 1쪽 = Document 1개 (28개) | 1교시 Loader |
| ② 정리 ★ | 쪽마다 붙은 머리말 «동양미래대학교 규정집 … 2-0-1~9» · «- 9 -» 제거, 줄 끝에서 끊긴 단어(«이⏎를») 잇기 | 1교시 «PDF 파싱 난점» |
| ③ 분할 | «제N조» 로 먼저 나누고 → 500자를 넘는 조만 재귀 분할 (`split_compare.py` 의 설정 그대로) | 2교시 2단 구성 |
| ④ 메타 | `source` · **`page`** · `chapter` · `article` — 부칙 · 별표 · 별지는 따로 표시 | 2교시 2절 |
| ⑤ 저장 | `qwen3-embedding:0.6b` → FAISS → `faiss_io.save_index()` | 3교시 실습 3 |

결과: **119 청크 · 92 조** (평균 175자). 표(입학정원 · 성적 등급)는 숫자 나열로 풀린 채 둡니다(범위 밖).

> ★ **«부칙» 을 따로 표시하는 이유** — 학칙 뒤쪽에는 개정 때마다 붙은 부칙이 50개 넘게 있고,
> 그 안에서 «제1조(시행일)» 가 매번 다시 시작합니다. 표시하지 않으면 본문 제1조(목적)와 출처가 섞입니다.

## 준비 — 세 가지가 먼저 되어 있어야 합니다

| 확인 | 어떻게 | 안 되면 |
|---|---|---|
| ① 슬랙 토큰 | 9주차 `socket_hello.py` 에서 pong 이 온다 | 9주차 실습 1 |
| ② 모델 | `ollama list` 에 `gemma3:4b` · `qwen3-embedding:0.6b` | `ollama pull …` |
| ③ PDF | `../data/학칙.pdf` 가 있다 | 아래 다운로드 명령 (브라우저로 받으면 이름이 `2-0-1 학칙 ver.41(2024. 7. 1.).pdf` → `학칙.pdf` 로 바꾸기) |

```bash
pip install -r requirements.txt

# ③ 학칙 PDF 받기 (이 폴더에서)
Invoke-WebRequest "https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do" -OutFile "..\data\학칙.pdf"   # PowerShell
curl -L -o ../data/학칙.pdf "https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do"                        # Git Bash
```

루트 `.env` (저장소 맨 위, 9주차와 같은 파일) 에 있어야 할 값:

```
SLACK_BOT_TOKEN=xoxb-…          # 9주차 그대로
SLACK_APP_TOKEN=xapp-…          # 9주차 그대로
SLACK_ALLOWED_USERS=U0…         # (권장) openai 를 나만 쓰게
EMBED_MODEL=qwen3-embedding:0.6b   # ⚠️ 인덱스를 만들 때와 읽을 때 같아야 합니다
MODEL=gemma3:4b                 # (선택) 생성 모델
OPENAI_API_KEY=sk-…             # (선택) openai 로 물을 때만
```

## 실행 (이 폴더에서)

```bash
# 0단계 — 인덱스 만들기 (한 번만. PDF 나 EMBED_MODEL 을 바꾸면 다시)
python build_index.py --show        # 저장 없이 청크 · metadata 만 (임베딩 없음 · 즉시)
python build_index.py               # index_pdf/ 생성

# 1단계 — 슬랙 없이 RAG 만 확인 ★ 봇이 이상할 때 범인을 가르는 기준점
python rag_backend.py 검색          # 검색만 (LLM 없음 · 즉시)
python rag_backend.py               # 검색 + 생성
python rag_backend.py 비교          # RAG 없이 vs RAG 로 ★

# 2단계 — 봇 켜기 (끝낼 때는 Ctrl+C)
python slack_rag_bot.py
```

> ⚠️ **9주차 봇은 끄고 켜십시오.** 같은 앱 토큰(xapp)으로 두 봇이 동시에 붙어 있으면 이벤트가 둘 중 한쪽으로 갈립니다.
> 💡 경로에 한글이 있으면 `index_pdf/` 에 `index.faiss` 대신 **`index.bytes`** 가 생깁니다 — 정상입니다(`../faiss_io.py`).

## 슬랙에서 쓰는 법

| 메시지 | 하는 일 |
|---|---|
| `@봇 몇 주를 결석하면 제적되나요?` | 학칙에서 찾아 답 + 📚 근거 조항 · 쪽 (Ollama) |
| `@봇 openai 몇 주를 결석하면 제적되나요?` | 같은 방식, OpenAI (`SLACK_ALLOWED_USERS` 만) |
| `@봇 검색 몇 주를 결석하면 제적되나요?` | 어떤 조항이 뽑혔는지만 — 거리와 앞부분 |
| `@봇 비교 몇 주를 결석하면 제적되나요?` | **RAG 없이** 답한 것과 **RAG 로** 답한 것을 나란히 ★ |
| `@봇` (내용 없음) | 사용법 안내 |

## 실행 결과 (2026-11-05, 이 PC — GPU 없음, gemma3:4b · qwen3-embedding:0.6b)

**검색** — 다섯 질문 모두 정답 조항이 1위입니다 (거리, 작을수록 가깝다).

| 질문 | 1위 |
|---|---|
| 몇 주를 결석하면 제적되나요? | `1.053` 제29조(제적) · p.7 ✅ |
| 휴학은 최대 몇 학기까지 할 수 있나요? | `0.836` 제25조(휴학) · p.6 ✅ |
| 졸업하려면 몇 학점이 필요한가요? | `0.724` 제42조(졸업) · p.10 ✅ |
| 등록금을 내지 않으면 어떻게 되나요? | `1.101` 제54조(등록금) · p.13 ✅ |
| 학사경고는 언제 받나요? | `0.993` 제41조(학사경고) · p.10 ✅ |

> 📌 아래 세 질문은 3교시 `학칙.md`(제5·7장 정리본)로는 답할 수 없던 것입니다 — 근거 문서가 PDF 전체로 넓어진 효과입니다.

**비교** — 같은 모델, 같은 질문. 바뀐 것은 근거를 넣었느냐뿐입니다. ★★

| | 답 (요약) |
|---|---|
| RAG 없이 | «5회 이상 결석 시 학사일지에 기록 … **20회 이상 결석 시 제적**» ❌ 지어냄 |
| RAG 로 (41초) | «**연속하여 4주를 초과**하여 결석한 경우 제적 … **제29조**» ✅ — 근거 `제29조(제적) · 학칙.pdf p.7` |

**졸업 학점** (94초) — «전문학사 2년 75학점 · 3년 110학점, 전공심화 2년 55학점 · 1년 20학점» ✅ (PDF 10쪽 제42조 ①②와 같음)

**슬랙 봇** — 실제 Socket Mode 접속(`Bolt app is running!`) 확인.
멘션 처리는 가짜 슬랙 client 로 확인했습니다(자리 잡기 «⏳ 학칙에서 찾는 중…» → 답으로 고치기).

- 이 PC 는 GPU 가 없어 한 답에 40~90초입니다. 그래서 봇은 먼저 «⏳» 를 달고, 답이 오면 그 메시지를 고칩니다(9주차와 같음).
- 모델이 근거 밖 내용을 섞을 때가 있습니다(위 결석 답의 «징계 위반 시에도…» 문장). 그래서 **📚 근거** 를 함께 보여 줍니다.

## LangSmith 에서 볼 것

- 슬랙 질문 1건 = trace 1건 (`slack-rag (ollama)`), 태그 `week10-slack-rag` · `rag`
- 그 안에 **Retriever(검색)** 와 **ChatOllama(생성)** 가 나란히 보입니다.
  답이 틀렸을 때 «검색이 틀렸나(엉뚱한 조항) / 생성이 틀렸나(조항은 맞는데 답이 틀림)» 를 여기서 가릅니다 ★
- `비교` 의 RAG 없는 쪽은 태그 `no-rag`

## 내 프로젝트로 바꾸기

| 바꿀 것 | 어디서 |
|---|---|
| 내 PDF 로 | `python build_index.py 내문서.pdf` — «제N조» 구조가 없으면 **쪽 단위**로 나눕니다 |
| 머리말 · 쪽 번호 규칙 | `build_index.py` 의 `HEADER` 정규식 — 내 PDF 의 머리말에 맞춘다 ★ |
| 근거 개수 | `rag_backend.py` 의 `TOP_K` |
| 봇의 말투 · 규칙 | `rag_backend.py` 의 `RAG_PROMPT` (⚠️ `<근거>` 구분자와 «지시가 아니다» 문장은 남긴다) |
| 출처 표시 | `rag_backend.py` 의 `source_of()` — 분할 때 심은 metadata 를 쓴다 |
| MCP 도구를 붙이려면 | `slack_rag_bot.py` 의 `parse()` 에 모드를 하나 더 두고, 그 모드에서 9주차 `mcp_test/mcp_client.py` 방식으로 답한다. ⚠️ `gemma3:4b` 는 도구 호출 미지원 → `llama3.2:3b` |

## 안 될 때

| 증상 | 원인 | 조치 |
|---|---|---|
| `인덱스가 없습니다` | 0단계를 안 했다 | `python build_index.py` |
| 시작할 때 `인덱스 검색 실패` | Ollama 꺼짐 / `EMBED_MODEL` 이 인덱스와 다름 | `ollama serve` · `.env` 확인 → 필요하면 `build_index.py` 다시 |
| `SLACK_BOT_TOKEN 은 xoxb-…` | 루트 `.env` 에 토큰 없음 | 9주차 실습 1 |
| 봇이 가끔만 답한다 | 9주차 봇이 같은 토큰으로 켜져 있다 | 9주차 봇 Ctrl+C |
| 답은 오는데 틀린 조항 | 검색 문제 | `@봇 검색 …` 으로 1위 확인 → 분할 · 임베딩을 의심 (10주차의 결론) |
| 조항은 맞는데 답이 틀림 | 생성 문제 | `RAG_PROMPT` 강화 · `openai` 로 비교 |
| 출처가 «부칙 · p.19» 로 나온다 | 질문이 개정 경과 조치와 가깝다 | 정상 — 본문 조항은 2 · 3위에 있는지 `검색` 으로 확인 |
| 한글이 `UnicodeEncodeError` | 콘솔 cp949 | `chcp 65001` 또는 `set PYTHONUTF8=1` |
