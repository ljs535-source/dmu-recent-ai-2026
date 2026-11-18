# 4주차 실습 코드 — Model I/O (1) 로컬 LLM과 상용 LLM을 하나의 인터페이스로

2026년 9월 18일 (금) / 최신인공지능 4주차

> **"코드 한 줄만 바꿉니다. 나머지는 그대로입니다."**
> 3주차에 만든 체인에서 **모델 자리만** 교체합니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| 셋팅 | 1교시 전 | [`setup_env.bat`](setup_env.bat) | **PC 초기화 복구** — venv + `.gitignore` + (2026-09-21 수정) **2~4주차 패키지(`langchain-openai==1.6.2`)** + (2026-09-21 추가) **[5/5] Git 설정**(이름 · 메일 · 저장소 주소 입력 → `--local`) + 이후 명령 안내 |
| 셋팅 (Git Bash) ★ | 1교시 전 | [`setup_env.sh`](setup_env.sh) | **VS Code 터미널을 Git Bash 로 열고 `bash setup_env.sh`** — venv · `.gitignore` · (2026-09-21 수정) **2~4주차 패키지(`langchain-openai==1.6.2` 포함)** · (2026-09-21 추가) **[5/5] Git 설정**(`git init` + 이름 · 메일 · 저장소 주소를 입력받아 `--local` 로 지정)까지 합니다. 남은 명령(VS Code · `pull`/`push` · `.env` · Ollama)만 Git Bash 명령으로 안내 |
| 확인 B | 1교시 | [`vram_calc.py`](vram_calc.py) | VRAM 요구량 어림 계산 (손계산 검산용) |
| **실습 0** ★ | 1교시 | [`Modelfile`](Modelfile) | **나만의 챗봇 만들기** — `ollama create py-tutor -f Modelfile` |
| 실습 0-③ ★★ | 1교시 | [`my_bot.py`](my_bot.py) | 만든 챗봇을 `ChatOllama` 로 부르기 — **코드 설정이 Modelfile 을 이긴다** |
| — | 2교시 2절 | [`bind_params.py`](bind_params.py) | `.bind()` 틀려 보기 · 공급자별 이름 — 조용히 무시되는 것 실측 (`max_tokens=10` vs `num_predict=10`, OpenAI 쪽 ③⑤ 는 키가 있을 때만) |
| **실습 1-①** ★★ | 2교시 | `first_chain.py` (3주차에서 복사) | **내 체인에서 `llm =` 한 줄 교체** — `ChatOllama` → `ChatOpenAI` |
| **실습 1-②** ★★ | 2교시 | [`swap_models.py`](swap_models.py) | **같은 체인 · 과제 3종 × 모델 2종** — 정답 확인 · 속도 분해(로딩 뺀 값 · 토큰/초) · 한 달 비용(`cost` 모드) |
| **실습 2** ★ | 2교시 | [`비교표_양식.md`](비교표_양식.md) | 6기준 비교표 작성 — 표 A 정답 확인 · 표 B 6기준, 칸마다 [측정]·[계산]·[판단] (저장소 커밋) |
| — | 3교시 | [`common_params.py`](common_params.py) | `temperature`·`max_tokens`·`timeout`·`max_retries` |
| **실습 3** | 3교시 | [`fallback.py`](fallback.py) | 상용 실패 → 로컬 대체 (`with_fallbacks`) |
| **실습 4** ★ | 3교시 | [`streaming.py`](streaming.py) | 첫 토큰 지연 측정 · invoke vs stream 비교 |

> 학생 저장소에서는 `week04/` 경로에 두게 합니다.
> 1교시의 `ollama list`·`show`·`ps` 는 터미널에서 직접 칩니다.
> **실습 0**(나만의 챗봇)은 `code` 폴더에서 `ollama create` 로 만들고, `my_bot.py` 로 코드에서 불러 봅니다.

## 사전 준비

> **실습실 PC가 초기화되었다면 먼저** `setup_env.bat` 을 실행하십시오 (탐색기에서 더블클릭).
> venv 생성 · 2~3주차 패키지 설치를 자동으로 하고, 이후 할 명령(VS Code · Git 복구 · `.env`)을 화면에 출력합니다.
> 자세한 내용은 1교시 PPT 2·3장 · 배포 1교시 0절.
> Git Bash 를 쓴다면 같은 일을 하는 `setup_env.sh` 를 Git Bash 에서 `bash setup_env.sh` 로 실행합니다.
> (추가) `setup_env.sh` 의 받기 · 질문별 답 · 끝난 뒤 할 일 · 오류 대처는 [**`setup_env_README.md`**](setup_env_README.md) 에 정리했습니다.
> (2026-09-14 추가) 두 스크립트는 프로젝트 폴더에 `.gitignore` 도 만들어 둡니다(이미 있으면 그대로). 3주차 저장소를 `git pull` 로 되살릴 때는 그 파일이 있으면 `pull` 이 멈춥니다.
> (2026-09-22 수정) **지우지 말고 옮깁니다** — `Rename-Item .gitignore .gitignore.bak`(Git Bash: `mv .gitignore .gitignore.bak`) → `git pull` → 성공하면 `.bak` 삭제, **실패하면 되돌리기**. 화면 안내 [4-A] 에 있습니다.
> 지웠다가 `pull` 이 실패하면 `.gitignore` 없이 남고, 그 상태의 `git add .` 은 `.env`(공용 키)와 `venv/`(실측 8,822개 파일)를 스테이징합니다.
> (2026-09-21 추가) **두 스크립트 모두 4단계가 아니라 5단계입니다** — `langchain-openai` 까지 설치하고, 이어서 `git init` · `user.name` · `user.email` · `origin` 을 **입력받아** 지정합니다.
> 즉 아래 «2) 4주차 패키지 설치» 와 화면 안내 [4] 의 git 설정 줄은 `.sh` 로 셋팅했다면 **이미 끝난 상태**입니다. 남는 것은 `pull` 또는 첫 `push` 뿐입니다(계정 인증이 필요해 자동으로 하지 않습니다).

```bash
# 1) 3주차 저장소(과제 1)를 clone 하고 가상환경 활성화
#    Windows
venv\Scripts\Activate.ps1
#    Git Bash  (setup_env.bat 으로 만든 venv 도 그대로 켜집니다)
source venv/Scripts/activate

# 2) 4주차 패키지 설치 — langchain-openai 가 추가됩니다
pip install -r requirements.txt

# 3) .env 준비  ★  (수업에서는 2교시 §1 에서 맨 먼저 — 키는 2교시 시작 직후 배포)
copy .env.example .env        # Windows  (macOS/Linux: cp)
#    → .env 의 OPENAI_API_KEY= 뒤에 수업 중 배포되는 공용 키를 넣습니다

# 4) ⚠️ 실습 시작 전 반드시 확인 — .env 가 여기 보이면 안 됩니다
git status

# 5) 모델 확인 (pull 은 수업 중 금지 ★ — 사전 설치되어 있습니다)
ollama list          # gemma3:4b 가 보여야 함
```

## 실행 순서 (수업 진행 순서)

```bash
# ── 1교시 ── 터미널에서 직접
ollama show gemma3:4b        # 파라미터 수 · 양자화 조회 ★ (context length 는 최대치)
python vram_calc.py          # 손계산과 맞춰 본다
python vram_calc.py 4.3 4096 # 내 화면의 값으로 계산 (컨텍스트는 실제 창 = ps 의 CONTEXT)
ollama run gemma3:4b "안녕"  # 메모리에 올리고
ollama ps                    # SIZE · PROCESSOR · CONTEXT 열 확인 ★
ollama stop gemma3:4b        # 2교시 전 VRAM 확보

# ── 1교시 §3 ── 실습 0  나만의 챗봇  ★
ollama show gemma3:4b --modelfile   # 원본의 레시피 카드
ollama create py-tutor -f Modelfile  # 만들기 (1초 · 디스크 약 1KB)
ollama run py-tutor                  # 대화 (/bye)
ollama show py-tutor                 # Parameters · System 확인
python my_bot.py                     # 실습 0-③  코드 설정이 이긴다  ★★

# ── 2교시 ── week04 폴더 · (venv)
# §1  .env 키 먼저  ★  (bind_params.py ③⑤ 의 OpenAI 쪽 에러·경고는 키가 있어야 보입니다)
copy .env.example .env       # → git status 로 .env 가 안 보이는지 확인 → OPENAI_API_KEY= 뒤에 붙여넣기
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('OPENAI_API_KEY')))"   # True

# §2  .bind() · 공급자별 이름
python bind_params.py        # ② 줄을 bind(temperature=0.9) 로 고쳐 틀려 보기 → TypeError 확인 후 되돌리기

# §3-1  실습 1-①  내 체인에서 한 줄 바꾸기  ★★
copy ..\week03\first_chain.py first_chain.py
#   llm = ChatOllama(model=MODEL)  →  llm = ChatOpenAI(model="gpt-4o-mini")   ← 이 줄 하나만
#   + import 2줄: from langchain_openai import ChatOpenAI / from dotenv import load_dotenv
#   + load_dotenv()                (키가 없으면 llm = ChatOllama(model="gemma3:1b"))
python first_chain.py

# §3-2  실습 1-②  같은 체인 · 과제 3종 × 모델 2종  ★★
python swap_models.py        # 한 번만 — 로컬 로딩 시간은 load_duration 으로 빼서 보여 줍니다

# §4-2  비용 — 1회에서 한 달로 (모델을 부르지 않고 계산만)
python swap_models.py cost 입력토큰 출력토큰 하루사용자 1인당호출   # 예: cost 60 150 300 5

# §4-3  실습 2  비교표_양식.md 작성 → git add · commit (.env 가 올라가지 않는지 확인)

# ── 3교시 ──
python common_params.py      # 1절
python fallback.py           # 실습 3
python streaming.py          # 실습 4  ★ → 첫 토큰까지 · 총 시간 측정
```

## 주의

- **`ollama pull` 은 수업 중 절대 실행 금지.** 30명이 동시에 수 GB를 받으면
  실습실 네트워크가 마비됩니다. 필요한 모델은 사전 pull 완료 상태입니다.
- 모든 파일 상단에 `LOCAL_MODEL = "gemma3:4b"` 상수를 두었습니다.
  실습실 모델이 다르면 **이 한 줄만** 고치면 됩니다.
- 🔶 **상용 모델명은 자주 바뀝니다.** 각 파일의 `OPENAI_MODEL` 값을
  **수업 전날 공식 문서에서 확인**해 확정하고 배포하십시오.
- 🔶 `usage_metadata`(LangChain 이 통일한 토큰 수)는 공급자·버전에 따라 비어 있을 수 있습니다.
  `swap_models.py` 는 이때 «(측정 불가)» 를 찍고 비용을 계산하지 않습니다.
  학기 전에 2종 모두 찍어 보고, 비어 있으면 비교표에 **"측정 불가"** 로 적게 하십시오.
  속도 분해에 쓰는 `response_metadata`(공급자 원본)는 모양이 달라, 구간별 시간(`load_duration`·`eval_duration`)은
  Ollama 에만 있고 OpenAI 의 토큰/초는 전체 시간으로 나눈 **근사값(≈)** 입니다.
- 🔶 **단가·환율은 `swap_models.py` 상단 상수**입니다 — `PRICE_IN_PER_1M`(0.15) · `PRICE_OUT_PER_1M`(0.60) · `KRW_PER_USD`(1400).
  연습용 가정값이므로 **수업 전날 공식 가격표·환율로 확정**하십시오. 한 달 시나리오 기본값은
  `USERS_PER_DAY`·`CALLS_PER_USER`·`DAYS_PER_MONTH`(100·10·30), 최대 출력은 `MAX_OUT`(400) 입니다.
- `swap_models.py` 과제 ③(추론)은 **악수 문제(정답 11)** 로 정했습니다 — 자료 제작 PC 5회 실측에서 gemma3:4b 2/5 · gemma3:1b 0/5.
  🔶 OpenAI 쪽은 수업 전 공용 키로 한 번 돌려 11 이 나오는지 확인합니다.
- **`.env` 는 절대 커밋하지 않습니다.** `.env.example` 만 커밋합니다.
- **`Modelfile` 은 확장자가 없습니다.** 메모장으로 저장하면 `Modelfile.txt` 가 되기 쉽고, 그러면
  `Error: no Modelfile or safetensors files found` 가 납니다. **VS Code · UTF-8** 로 저장하십시오.
  (UTF-8 이 아니면 **오류 없이** 한글 SYSTEM 이 깨진 채 만들어집니다 — 2026-09-12 실측)
- Modelfile 의 `FROM` 에 **PC에 없는 모델**을 적으면 `create` 가 내려받기(`pulling manifest`)를 시작합니다. 즉시 Ctrl+C.

## ⚠️ Claude(Anthropic)는 제외입니다

`ChatAnthropic` 은 **Anthropic API 키**가 필요하며, 학생이 가진
**Claude Code 구독 계정으로는 호출할 수 없습니다.**
교체 실습은 **로컬 ↔ OpenAI 2종**으로 진행하며, `langchain-anthropic` 도 설치하지 않습니다.

## 상용 키를 못 쓰게 된 경우 (자동 대체)

`swap_models.py` 와 `streaming.py` 는 `OPENAI_API_KEY` 가 없으면
**로컬 2종 비교**(`gemma3:1b` vs `gemma3:4b`)로 자동 전환됩니다. 코드 구조는 그대로입니다.

| 실습 | 대체 방식 |
|---|---|
| 2교시 2절 | `bind_params.py` 의 ③·⑤(OpenAI 쪽)는 «건너뜀» — `ChatOpenAI` 는 키가 없으면 객체를 만들 때부터 막힘. ②④⑥ 은 그대로 |
| 실습 1-① | `first_chain.py` 의 `llm =` 줄을 `ChatOllama(model="gemma3:1b")` 로 — 한 줄 교체는 똑같이 해 봄 |
| 실습 1-② | 로컬 2종 비교 (자동) — 비용은 «(가정) 상용 단가였다면» 참고값 |
| 실습 2 | 비교표의 **'비용' → '메모리 점유'**(`ollama ps` 의 SIZE)로 치환 |
| 실습 3 | `fallback.py` 의 `primary` 를 `ChatOllama(model="없는모델")` 로 바꿔 로컬→로컬 폴백 |
| 실습 4 | 변경 없음 (로컬에서도 스트리밍 동작) |

## 오늘 산출물 확인

- [ ] `my-bot.Modelfile` — 나만의 챗봇 (실습 0)
- [ ] `first_chain.py` — `llm =` 한 줄 교체 후 실행 (실습 1-①)
- [ ] `swap_models.py` — 과제 3종 × 모델 2종 측정 + `cost` 로 한 달 비용 계산 (실습 1-②)
- [ ] `fallback.py` — 폴백 동작 확인
- [ ] `streaming.py` — 첫 토큰 지연 측정
- [ ] **비교표 (6기준 완성본 — 표 A 정답 확인 · 표 B 6기준)** ★ — 저장소에 커밋

> 📌 비교표는 정규 과제가 아니지만 버리지 마십시오.
> **10주차 미니 프로젝트 명세**에서 *"왜 이 모델을 골랐는가"* 의 근거로 그대로 씁니다.
