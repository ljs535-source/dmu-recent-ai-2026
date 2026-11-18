# setup_env.sh 사용 설명서 — 실습 환경 한 번에 만들기 (Git Bash)

> 실습실 PC 가 **초기화되었거나**, 새 PC 에서 처음 시작할 때 실행합니다.
> 2~4주차 실습에 필요한 **폴더 · 가상환경 · 패키지 · Git 설정**을 한 번에 만들어 줍니다.

같은 일을 하는 Windows 판이 [`setup_env.bat`](setup_env.bat) 입니다(탐색기에서 더블클릭).
**Git Bash 를 쓰면 이 `setup_env.sh` 를 쓰세요.** 둘 중 하나만 실행하면 됩니다.

---

## 1. 스크립트가 해 주는 것 / 내가 해야 하는 것

| 단계 | 스크립트가 해 주는 것 | 내가 입력하는 것 |
|---|---|---|
| [0] 사전 점검 | Python 3.10 이상 · git · ollama · `gemma3:4b` · VS Code 가 있는지 확인 | — |
| [1/5] 작업 위치 | 프로젝트를 만들 **상위 폴더**로 이동 (없으면 만들지 물어봄) | 상위 폴더 경로 |
| [2/5] 프로젝트 폴더 | 폴더 생성 + **`.gitignore`** 생성 (`venv/` · `.env` 가 GitHub 에 올라가지 않게) | 폴더 이름 |
| [3/5] 가상환경 | `python -m venv venv` → 활성화 | (이미 있으면) 새로 만들지 여부 |
| [4/5] 패키지 | 2~4주차 패키지 설치 → 버전 확인 | — |
| [5/5] Git 설정 | `git init` + 이름 · 메일 · 저장소 주소를 **이 폴더에만**(`--local`) 저장 | 이름 · 메일 · 저장소 주소 |

스크립트가 **하지 않는 것** — 끝나면 화면에 명령이 나오니 따라 치세요 ([5. 끝난 뒤](#5-끝난-뒤-직접-할-일-1--6)).

- VS Code 로 폴더 열기 · 인터프리터 지정
- `git pull` / `git push` (GitHub 로그인이 필요해서)
- `.env` 만들기 (키는 수업에서 받습니다)
- Ollama 모델 받기

설치되는 패키지 (버전 고정):

```
langchain==1.4.0  langchain-core==1.6.2  langchain-ollama==1.1.0  langchain-openai==1.6.2
python-dotenv==1.2.3  ollama==0.6.2  numpy
```

---

## 2. 시작 전 준비물

- [ ] **Python 3.10 이상** · **Git**(Git Bash 포함) · **Ollama** · **VS Code** 가 설치된 PC — 실습실 PC 에는 모두 있습니다
- [ ] 3주차에 만든 **GitHub 저장소 주소** (예: `https://github.com/<본인계정>/langchain-2026.git`) — 없으면 나중에 넣어도 됩니다
- [ ] Git 에 쓸 **이름**과 **메일**
- [ ] 프로젝트를 둘 **짧은 영문 경로** (예: `D:\dl026`) — 경로가 길거나 한글이 있으면 뒤에서 문제가 생길 수 있습니다

---

## 3. 받기

이 파일은 강의 저장소의 `4week/code/setup_env.sh` 입니다. 아래 셋 중 편한 방법으로 받으세요.

```bash
# ① Git Bash 에서 바로 받기 (받을 폴더로 이동한 뒤)
curl -LO https://raw.githubusercontent.com/ljs535-source/dmu-recent-ai-2026/main/4week/code/setup_env.sh

# ② 강의 저장소를 이미 받아 두었다면 복사
cp <강의저장소>/4week/code/setup_env.sh ./
```

③ GitHub 웹에서 파일을 열고 «Download raw file» 버튼으로 받아도 됩니다.

> ⚠️ **메모장 등으로 열어 저장하지 마세요.** 줄바꿈이 Windows 식(CRLF)으로 바뀌면
> `$'\r': command not found` 오류로 실행되지 않습니다. 그때는 위 ① 로 다시 받으면 됩니다.

---

## 4. 실행

1. VS Code 를 열고 **터미널을 Git Bash 로** 엽니다 — 터미널 창 오른쪽 **`+` 옆 `∨` → Git Bash**
2. `setup_env.sh` 가 있는 폴더로 이동해 실행합니다.

```bash
bash setup_env.sh
```

> 💡 Git 을 기본 설정으로 설치했다면 탐색기에서 **더블클릭**해도 Git Bash 창에서 실행됩니다.
> 끝나면 «Enter 를 누르면 끝납니다...» 에서 멈추므로 결과를 읽고 Enter 를 누르세요.

### 질문에 이렇게 답하세요

| 단계 | 화면에 나오는 질문 | Enter 만 누르면 | 입력 예 · 주의 |
|---|---|---|---|
| [1/5] | `상위 폴더 경로` | **지금 폴더** | `D:\dl026` — 탐색기 주소창의 경로를 그대로 붙여 넣어도 됩니다(따옴표가 붙어 와도 괜찮음) |
| [1/5] | `이 폴더를 만들까요?` (없는 폴더일 때) | 만들기 | `n` = 경로 다시 입력 |
| [2/5] | `만들 폴더 이름` | **`langchain-2026`** | **영문 · 숫자 · `-` `_` `.` 만** (공백 · 한글 불가) |
| [2/5] | `이 위치로 진행할까요?` | 진행 | `n` = 이름 다시 입력 |
| [3/5] | `지우고 새로 만들까요?` (venv 가 이미 있을 때) | **그대로 사용** | `y` = 지우고 새로 (설치가 꼬였을 때) |
| [5/5] | `이름 (git user.name)` | 이전 값 (있으면) | `홍길동` — 비우면 다시 물어봅니다 |
| [5/5] | `메일 (git user.email)` | 이전 값 (있으면) | `hong@example.com` — 메일 모양이 아니면 다시 물어봅니다 |
| [5/5] | `저장소 주소` | **나중에** (또는 이전 값) | `https://github.com/<본인계정>/langchain-2026.git` |

### 실행 화면 (예)

```text
$ bash setup_env.sh
============================================================
 최신인공지능 — 실습실 PC 초기 셋팅 (Git Bash)
============================================================

[0] 사전 점검
  [OK] Python 3.13.7  (python)
  [OK] git
  [OK] ollama
  [OK] gemma3:4b
  [OK] VS Code
  ※ [X] 가 있으면 조교/교수에게 알려 주세요.

[1/5] 작업 위치
  프로젝트 폴더를 만들 «상위 폴더» 를 입력하세요.
  탐색기 주소창의 경로를 복사해 붙여 넣어도 됩니다 (예: D:\dl026).
  상위 폴더 경로 (Enter = 지금 폴더 C:\Users\student): D:\dl026
  위치: D:\dl026

[2/5] 프로젝트 폴더
  만들 폴더 이름 (Enter = langchain-2026):
  만들 위치: D:\dl026\langchain-2026
  이 위치로 진행할까요? (Enter = 진행 / n = 이름 다시 입력):
  [OK] .gitignore 생성 — venv/ · .env 가 git 에 올라가지 않게

[3/5] 가상환경
  python -m venv venv  (10~30초)
  [OK] 활성화됨: D:\dl026\langchain-2026\venv\Scripts\python.exe

[4/5] 패키지 설치 (2~4주차)
  pip install langchain==1.4.0 langchain-core==1.6.2 ...
  (설치 로그 — 1~3분)
  설치 확인:
    langchain          1.4.0
    langchain-core     1.6.2
    ...

[5/5] Git 설정
  [OK] git init
  이름 (git user.name): 홍길동
  메일 (git user.email): hong@example.com
  GitHub 저장소 주소 — 3주차에 만든 저장소가 있으면 그 주소를 붙여 넣으세요.
  저장소 주소 (Enter = 나중에): https://github.com/hong/langchain-2026.git
  [확인] user.name  = 홍길동
         user.email = hong@example.com
         origin     = https://github.com/hong/langchain-2026.git

============================================================
 설치 완료 — 아래 명령은 직접 실행하세요
============================================================
```

`[0] 사전 점검` 에서 표시의 뜻:

| 표시 | 뜻 |
|---|---|
| `[OK]` | 준비됨 |
| `[--]` | 없어도 진행은 됩니다 (예: Ollama 가 꺼져 있어 `gemma3:4b` 를 확인 못 함) |
| `[X]` | **필요한 것이 없습니다** — 조교/교수에게 알려 주세요 |

---

## 5. 끝난 뒤 직접 할 일 [1] ~ [6]

> ⚠️ **스크립트 안에서 켠 가상환경은 스크립트가 끝나면 꺼집니다.** 아래 [2] 에서 다시 켜세요.
> 화면에 나온 안내와 같은 내용입니다. 폴더 경로는 화면에 나온 내 경로를 쓰세요.

**[1] VS Code 로 프로젝트 폴더 열기** — 이미 그 폴더를 VS Code 로 열어 두었다면 건너뜁니다.

```bash
code "/d/dl026/langchain-2026"
```

**[2] 가상환경 켜기** — 프롬프트 위 줄에 **`(venv)`** 가 붙어야 정상입니다.

```bash
cd "/d/dl026/langchain-2026"
source venv/Scripts/activate
which python                  # .../venv/Scripts/python 이면 OK
```

- `which python` 이 venv 가 아니면: `export PATH="$PWD/venv/Scripts:$PATH"`
- PowerShell 에서는: `venv\Scripts\Activate.ps1` — Git Bash 로 만든 venv 도 그대로 켜집니다

**[3] VS Code 인터프리터 지정** — `Ctrl+Shift+P` → **Python: Select Interpreter** → `./venv/Scripts/python.exe`

공용 PC 라면 설정을 폴더 안에 남겨 두면 다음에 편합니다 (한 줄씩 실행).

```bash
mkdir -p .vscode
echo '{ "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe" }' > .vscode/settings.json
```

**[4] Git** — 이름 · 메일 · 저장소 주소는 [5/5] 에서 끝났습니다. 확인만 하세요: `git config --local --list`

저장소 주소를 나중에 넣으려면: `git remote add origin https://github.com/<본인계정>/langchain-2026.git`

그다음 **둘 중 하나**를 고릅니다.

| 내 상황 | 할 일 |
|---|---|
| **[4-A] 3주차 저장소가 GitHub 에 이미 있다** ★ 대부분 | 내 저장소 내용을 받아 온다 (아래 4-A) |
| **[4-B] 저장소가 없다 (처음부터)** | 지금 폴더를 첫 커밋으로 올린다 (아래 4-B) |

**[4-A] 저장소가 이미 있는 경우**

```bash
mv .gitignore .gitignore.bak  # 지우지 말고 옮겨 둡니다 (그냥 두면 pull 이 멈춥니다)
git pull origin main
#   성공하면 →  rm .gitignore.bak
#   실패하면 →  mv .gitignore.bak .gitignore   ★ 반드시 되돌리고 나서 다시 시도
git branch -u origin/main
git status                    # venv/ 와 .env 가 보이면 안 됩니다 — 보이면 commit 하지 말고 손 드세요
```

> 왜 `.gitignore` 를 옮기나요? 스크립트가 만든 `.gitignore` 와 저장소의 `.gitignore` 가 **내용이 같아도**,
> 아직 git 이 모르는 파일이 같은 이름으로 있으면 `git pull` 이 «would be overwritten» 오류로 멈춥니다.
> 실패했을 때 되돌리지 않으면 `.env` 와 `venv/` 를 막아 줄 파일이 사라지니 꼭 되돌리세요.

**[4-B] 저장소가 없는 경우**

```bash
cat .gitignore                # venv/ · .env 가 들어 있는지 확인
pip freeze > requirements.txt
git add .
git status                    # venv/ 와 .env 가 없어야 합니다 — 보이면 commit 하지 말고 손 드세요
git commit -m "chore: 프로젝트 초기 세팅"
git push -u origin main       # 처음이면 GitHub 로그인 창이 뜹니다
```

**[5] `.env` 준비** — 키는 수업 중에 받습니다.

```bash
cp .env.example .env
git status                    # .env 가 목록에 보이면 안 됩니다 ★
```

**[6] Ollama 모델 확인** — `ollama list` 에 `gemma3:4b` 가 보여야 합니다.
⚠️ 실습실에서는 **수업 중에 `ollama pull` 하지 마세요** (여럿이 동시에 받으면 네트워크가 멈춥니다).

---

## 6. 다시 실행해도 되나요?

**됩니다.** 이미 있는 것은 망가뜨리지 않고 다시 씁니다.

| 이미 있는 것 | 스크립트의 동작 |
|---|---|
| 프로젝트 폴더 | `[i] 이미 있는 폴더입니다 — 그대로 사용합니다.` |
| `.gitignore` | 건드리지 않습니다 |
| `venv` | **Enter = 그대로 사용**, `y` = 지우고 새로 만들기 |
| 패키지 | 다시 설치를 시도합니다 (이미 있으면 금방 끝남) |
| git 저장소 · 이름 · 메일 · 주소 | `git init` 은 건너뛰고, 질문에서 **Enter 로 이전 값**을 그대로 씁니다 |

> 💡 패키지 설치가 네트워크 문제로 끊겼다면 — 다시 실행하고 venv 질문에서 **Enter(그대로 사용)** 를 고르세요.

> Git 이름 · 메일을 `--global` 이 아니라 **`--local`(이 폴더에만)** 로 저장하는 이유:
> 실습실 PC 는 여럿이 함께 씁니다. `--global` 로 넣으면 다음 사람의 커밋에 내 이름이 찍힙니다.

---

## 7. 이럴 때는

| 화면에 나온 것 | 원인 | 이렇게 하세요 |
|---|---|---|
| `$'\r': command not found` | 파일 줄바꿈이 CRLF 로 바뀜 | 메모장으로 저장하지 말고 [3. 받기](#3-받기) ① 로 다시 받기 |
| `Python 을 찾을 수 없습니다` · Microsoft Store 가 열림 | Windows 의 «앱 실행 별칭» | 설정 → 앱 → 앱 실행 별칭 → **python 끄기** |
| `Python 3.10 이상이 필요합니다` | 옛 Python | 조교/교수에게 알려 주세요 |
| `폴더 이름은 영문·숫자·-·_·. 만 쓸 수 있습니다` | 공백 · 한글 | 영문으로 다시 입력 (예: `langchain-2026`) |
| `[!] 경로가 너무 깁니다 (120자 초과)` | Windows 경로 길이 제한 | `n` → 짧은 상위 폴더(예: `D:\dl026`)로 다시 |
| `가상환경을 지우지 못했습니다` · `만들지 못했습니다` | VS Code 터미널 등이 venv 를 쓰는 중 | 그 프로그램을 모두 닫고 다시 실행 |
| `가상환경 활성화에 실패했습니다` | venv 가 망가짐 | 다시 실행 → venv 질문에서 **`y`** |
| `패키지 설치에 실패했습니다` | 네트워크 · 경로 길이 | 화면 안내대로 다시 실행(venv 는 Enter), 경로가 길면 짧은 위치에 새로 |
| `[X] git 없음` · `[X] ollama 없음` | 프로그램 미설치 | 조교/교수에게 알려 주세요 |
| `[--] gemma3:4b 확인 안 됨` | Ollama 가 꺼져 있음 / 모델 없음 | Ollama 를 켜고 `ollama list` 로 확인 — 셋팅은 계속 진행됩니다 |
| `git pull` 이 `untracked working tree files would be overwritten` | 스크립트가 만든 `.gitignore` | [4-A] 대로 `mv .gitignore .gitignore.bak` 먼저 |
| `git status` 에 `venv/` 나 `.env` 가 보임 | `.gitignore` 가 없음 | **commit 하지 말고 손 드세요** |
| 한글이 `???` 로 깨짐 | Git Bash 의 출력 인코딩 | 스크립트는 자동으로 맞춥니다. 직접 실행할 때는 `export PYTHONIOENCODING=utf-8` |
