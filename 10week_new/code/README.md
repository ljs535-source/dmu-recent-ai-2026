# 10주차 실습 코드 — RAG (1) 적재 · 분할 · 임베딩 / 미니 프로젝트 착수

**2026년 11월 6일 (금)** / 최신인공지능 10주차

> **"검색 품질의 상한은 임베딩 모델이 아니라 분할이 정합니다."** ★★

> 📌 **10주차 실습은 이 폴더(`10week_new/code`)를 씁니다.** `10week/code` 는 이전 판입니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| **실습 1** | 3교시 1절 (1교시 코드 읽기) | [`loaders.py`](loaders.py) | **Loader 4종**(Text·PDF·CSV·Web) 적재 + 파싱 난점 비교 |
| **실습 2** ★★ | 3교시 2절 (2교시 코드 읽기) | [`split_compare.py`](split_compare.py) | **분할 3종 → 검색 결과 비교** — 이 차시의 핵심 |
| — | 3교시 2절 (2교시 2절 판서) | [`split_compare.py --meta`](split_compare.py) | ★★ **메타데이터 설계** (`build_chunks_with_metadata`) |
| **실습 3** | 3교시 3절 | [`embed.py`](embed.py) | 임베딩 적용 + **한국어 검증** + ★★ **인덱스 저장** |
| 공용 | — | [`faiss_io.py`](faiss_io.py) | 인덱스 저장 · 읽기 — **경로에 한글이 있어도** 되게 한다 (직접 실행하지 않음) |
| 실습 문서 | 1교시 · 3교시 | [`data/`](data/) | ★ **모두 같은 문서 1건** — 문서가 다르면 결과를 비교할 수 없습니다 |
| 템플릿 | 2교시 4절 | [`project_template/`](project_template/) | ★ 미니 프로젝트 `README.md` / `CONTEXT.md` |
| 견본 | — | [`.env.example`](.env.example) | `EMBED_MODEL` 이 **새로 추가**됩니다 |
| (추가) 슬랙 연동 | 미니 프로젝트 출발점 | [`slack_bot_mode/`](slack_bot_mode/) | ★ 9주차 슬랙 봇 + **학칙.pdf 전체** RAG (`build_index.py` → `index_pdf/`) — **슬랙으로 묻고 RAG 로 답한다** |

`data/` 의 문서는 실제 학칙(규정번호 2-0-1, 2024.07.01. 개정)입니다.

| 파일 | 내용 |
|---|---|
| `학칙.pdf` | 학칙 전체 28쪽 (원본 PDF) — ⚠️ **저장소에 없습니다.** [학교 규정집 다운로드 링크](https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do)에서 받아 `data/학칙.pdf` 로 넣으세요 (아래 «사전 준비») |
| `학칙.md` | 제5장 · 제7장을 옮긴 정리본 (머리말 · 쪽 번호 · 개정 표기 제거, 3,388자) — **실습 2 · 3 의 문서** |
| `규정.txt` · `faq.csv` | 같은 조문으로 만든 Text · CSV Loader 실습용 |

## ⚠️⚠️ 오늘 반드시 커밋하세요 ★★

```
   ① index_recursive/            ← 11주차 실습 1 · 13주차 실습 4 가 이어 씁니다
   ② README.md / CONTEXT.md      ← 미니 프로젝트 명세 2% 입니다
```

인덱스가 없으면 11주차를 **다시 적재하는 것부터** 시작해야 합니다.

## 사전 준비

```bash
# 9주차까지 쓰던 가상환경을 그대로 사용합니다
venv\Scripts\Activate.ps1          # Windows PowerShell
source venv/Scripts/activate       # Git Bash

pip install -r requirements.txt     # ★ pypdf · beautifulsoup4 · faiss-cpu 가 새로 추가됩니다

# ★ 학칙 PDF 받기 — 저장소에는 없습니다. 학교 규정집에서 받아 data/ 에 넣습니다 (code 폴더에서)
Invoke-WebRequest "https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do" -OutFile "data\학칙.pdf"   # PowerShell
curl -L -o data/학칙.pdf "https://www.dongyang.ac.kr/regltn/dongyang/2/1037/download.do"                      # Git Bash

# 루트 .env (저장소 맨 위) 에 .env.example 의 EMBED_MODEL 줄을 옮겨 적습니다

# ★★ 임베딩 모델 — 실습실 PC 에는 미리 받아 두었습니다. 개인 PC 라면 수업 전에 받아 오세요.
ollama pull qwen3-embedding:0.6b
ollama list                         # qwen3-embedding:0.6b 가 보여야 함
```

> 💡 `python faiss_io.py` 를 한 번 실행해 보세요. 저장소 경로에 한글이 있는지 알려 줍니다.
> 한글이 있어도 실습은 됩니다(`index.faiss` 대신 `index.bytes` 로 저장). 영문 경로(예: `C:\dev\langchain-2026`)가 가장 깔끔합니다.

## 실행 순서

```bash
# ── 실습 1 : 적재 (1교시에 코드 읽기)
python loaders.py                  # 로컬 파일 3종 (네트워크 없음)
python loaders.py --web            #   Web 까지 (⚠️ 조별로 차례대로)

# ── 실습 2 : 분할 ★★ 10주차의 핵심 (2교시에 코드 읽기)
python split_compare.py            # 1단계 — 어디서 끊겼는지 눈으로 (임베딩 없음 · 즉시) ★
python split_compare.py --search   # 2단계 — 같은 질문, 다른 결과 ★★
python split_compare.py --meta     # 3단계 — 메타데이터 설계 ★★

# ── 실습 3 : 임베딩 + 저장
python embed.py --check            # 한국어 검증 ★★
python embed.py                    # 임베딩 + 인덱스 저장 ★★ 반드시 커밋
```

## ★★ 실습 2 에서 확인할 것

질문 «**몇 주를 결석하면 제적되나요?**» (정답: 학칙 제29조 2호 «연속하여 4주를 초과»)를
A 고정 · B 재귀 · C 구조 세 인덱스에 똑같이 던집니다. **2교시에 적어 둔 예측과 맞춰 보세요.**

| 분할 | 청크 수 | 평균 길이 | 검색 1위 청크에 **답이 들어 있는가** |
|---|---|---|---|
| A 고정 | | | |
| B 재귀 | | | |
| C 구조 | | | |

- `python split_compare.py` 출력에서 **A 의 청크가 어디서 끊겼는지** 먼저 보세요 — 단어 중간? 조 경계를 넘어서?
- 💡 `.env` 의 `WEEK10_QUESTION` 으로 질문을 바꿔 보세요. 표현에 따라 결과가 달라집니다.
- 📌 **바꾼 것은 자르는 방법뿐입니다.** 같은 임베딩 모델, 같은 질문입니다.

## 코드 읽기 포인트 — 슬라이드 코드와 다른 곳

**① `page` 키가 없습니다** ★

```python
# 슬라이드 (2교시 2-2)
"page": page_of(i),
    → .md 문서에는 애초에 페이지 개념이 없습니다.

# 이 폴더의 코드 (split_compare.py)
    → page 키를 아예 넣지 않습니다.
      ⚠️ 값이 없다고 None 을 넣으면 안 됩니다 — Chroma 등 일부 저장소가 metadata 의 None 을 거부합니다.
```

> 📌 *"페이지 번호는 PDF Loader 라야 붙습니다."* — 설계 원칙 ①(**나중에 복원할 수 없는 것을 우선**)의 실물 사례입니다.
> `slack_bot_mode/` 는 `학칙.pdf` 를 쓰므로 답에 «p.7» 처럼 쪽 번호가 붙습니다. 비교해 보세요.

**② 메타데이터를 '2단 구성' 으로 심습니다**

재귀 분할(B) 결과에 메타데이터를 손으로 붙이면 **장 · 조 정보를 알 방법이 없습니다**(재귀 분할은 구조를 모릅니다).

```
   이 폴더의 코드:  C(구조)로 먼저 나눠 장·조를 얻고 → 긴 섹션만 B(재귀)로 다시
                    = 1-4절에서 "실무에서 흔하다" 고 소개한 바로 그 2단 구성 ★
```

**③ `--search` / `--meta` 를 나눴습니다**

`split_compare.py` 는 옵션 없이 돌리면 **임베딩 호출 없이 즉시** 끝납니다.
«어디서 끊겼는가» 를 먼저 눈으로 보고, 검색 비교는 그다음입니다. ★

## ⚠️ 안 될 때

| 증상 | 원인 | 조치 |
|---|---|---|
| `qwen3-embedding:0.6b` not found | 모델 미설치 | `ollama pull qwen3-embedding:0.6b` (⚠️ 실습실에서 여럿이 동시에 받지 마세요) |
| **한국어 유사도가 셋 다 비슷** | 한국어를 못 잡는 모델 | `embed.py --check` 가 경고합니다 → **모델 교체** |
| `embed.py --check` 가 "실패" 를 출력 | 판정선(margin 0.03)이 엄격합니다 | 같은 의미 쌍이 다른 의미 쌍보다 **높은지** 직접 숫자를 보세요 |
| **세 분할의 검색 결과가 같음** | 질문이 청크 경계에 안 걸림 | 질문을 바꾸거나 `chunk_size` 를 바꿔 보세요 |
| `MarkdownHeaderTextSplitter` 결과가 1개 | 문서에 `#` 구조가 없음 | 구조 기반은 **구조가 있어야** 씁니다 |
| PDF 를 건너뜀 | `data/학칙.pdf` 없음 | «사전 준비» 의 다운로드 명령 — 브라우저로 받았다면 파일 이름을 `학칙.pdf` 로 바꿔 `data/` 에 |
| `faiss` import 오류 | 미설치 | `pip install faiss-cpu` |
| `save_local() 이 실패해 바이트 방식으로 저장` | 경로에 한글 | 정상입니다 — `faiss_io.py` 가 대신 저장합니다 |
| Web Loader 차단 / 타임아웃 | 동시 호출 | 조별로 차례대로 / `--web` 생략 |
| 한글이 `UnicodeEncodeError` | 콘솔 코드페이지(cp949) | `chcp 65001` 또는 `set PYTHONUTF8=1` |

## 미니 프로젝트 (2교시 안내 · 수업 후 작성) ★★

| 항목 | 비중 | 시점 |
|---|---|---|
| **명세 문서의 충실도** | **2%** | **10주차** |
| 구현 완성도 | 4% | 14주차 |
| 문서화 품질 (README) | 2% | 14주차 |

- 템플릿: [`project_template/README.md`](project_template/README.md) · [`project_template/CONTEXT.md`](project_template/CONTEXT.md)
- 필수 요건
  - **구성: 슬랙으로 묻고 슬랙으로 답한다** + 처리는 **RAG 또는 MCP** (하나 이상) — 출발점은 [`slack_bot_mode/`](slack_bot_mode/)
  - **LangChain 구현 + LangSmith 추적**
  - **VRAM 8GB 에서 실제로 동작**
- ★ **발표가 없습니다. README 가 곧 점수입니다.**

> ⚠️ **명세서(`README.md` 1~4번 + `CONTEXT.md`)는 3교시 실습을 마친 뒤 작성합니다.** 마감과 제출처는 수업에서 공지합니다.

## 꼭 이해해야 할 지점

- **RAG 파이프라인 6단계** (Load → Split → Embed → Store → Retrieve → Generate) ★
- ★ **파인튜닝과 RAG 의 선택 기준** — 지식 갱신 주기 / 비용 / **출처 추적 가능성**
- 문서 형식별 **파싱 난점** (표 · 머리말 · 페이지 번호 · 스캔본)
- `Document` 의 두 구성요소와 `metadata` 의 용도
- **분할 전략 3종**의 차이와 각각이 적합한 상황 ★
- ★★ **청크 크기와 오버랩의 트레이드오프** — 작을 때/클 때 각각 무엇이 나빠지는가
- ★★ **메타데이터 설계를 분할 단계에서 해야 하는 이유**
- **임베딩 모델 선택 기준** (한국어 지원 · 차원 · 속도 · 로컬 실행) ★
- 임베딩 모델의 **한국어 성능을 검증하는 방법**
- (4주차 연계) 생성 모델과 임베딩 모델을 **함께 올릴 때의 VRAM 계산**

## 오늘의 한 줄

> **쓰레기를 넣으면 쓰레기가 나온다.**
>
> 그리고 — **검색 품질의 상한은 임베딩이 아니라 분할이 정한다.** ★★
