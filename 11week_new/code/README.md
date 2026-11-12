# 11주차 실습 코드 — RAG (2) 벡터 저장소 · 검색 전략 · Lost in the Middle

**2026년 11월 13일 (금)** / 최신인공지능 11주차

> **"생성 모델을 더 좋은 것으로 바꾸는 것보다 검색을 개선하는 편이 효과가 큽니다."**
> — 이 말을 오늘 실험으로 확인합니다.

> 📌 **11주차 실습은 이 폴더(`11week_new/code`)를 씁니다.** 10주차 `10week_new/code` 의 인덱스를 이어 씁니다.

## 파일 구성

| 실습 | 교시 | 파일 | 내용 |
|------|------|------|------|
| 공용 | — | [`rag_common.py`](rag_common.py) | ★ **10주차 인덱스를 이어 쓰는 잡일** (없으면 다시 만듦) — 단독 실행하면 점검 |
| 공용 | — | [`faiss_io.py`](faiss_io.py) | 인덱스 읽기 · 저장 — 경로에 한글이 있어도 되게 한다 (10주차와 같은 파일) |
| 질문셋 | 2·3교시 | [`questions.py`](questions.py) | ★ **모두 같은 질문 10개** — LLM 호출 없음 · 무료 |
| **실습 1-①** | 1교시 판서 · 3교시 실습 | [`vector_stores.py`](vector_stores.py) | ★ **FAISS 와 Chroma 나란히** — 정체 · 점수 · 거리 척도 · metadata 필터 |
| **실습 1-②** | 1교시 판서 · 3교시 실습 | [`rag_chain.py`](rag_chain.py) | **LCEL RAG 체인** `retriever \| prompt \| llm \| parser` + `search_type` 3종 |
| **실습 2** ★★ | 2교시 판서 · 3교시 실습 | [`retriever_strategies.py`](retriever_strategies.py) | **MMR · MultiQuery · 하이브리드** — 같은 질문셋으로 비교표 |
| 읽기 자료 | 2교시 | [`concepts_4.py`](concepts_4.py) | 개념 4종 + **7종 한눈에 표** (호출 없음) · `--run` 으로 2종 체험 |
| **실습 3** ★ | 2교시 판서 · 3교시 실습 | [`lost_in_middle.py`](lost_in_middle.py) | **정답 문서의 위치 · 개수 실험** — 모델을 바꿔 비교 |
| **실습 4** | 2교시 판서 · 3교시 실습 | [`citation.py`](citation.py) | **출처 표기** + 인용 번호 검증 / `--followup` 후속 질문 다시 쓰기 |
| 견본 | — | [`.env.example`](.env.example) | 인덱스 위치를 바꿨을 때만 필요 |

## 사전 준비

```bash
# 10주차까지 쓰던 가상환경을 그대로 사용합니다
venv\Scripts\Activate.ps1          # Windows PowerShell
source venv/Scripts/activate       # Git Bash

pip install -r requirements.txt     # ★ langchain-chroma · chromadb · rank_bm25 · langchain-classic 이 새로 추가

python rag_common.py                # ★ 10주차 인덱스 점검 — 청크 14개 · 1024차원이 보이면 정상
```

- 10주차 `index_recursive/` 가 없으면 `rag_common.py` 가 학칙.md 로 그 자리에서 다시 만듭니다(⏱ 1~2분).
- 실행하면 `langchain-community is being sunset …` 경고가 한 줄 나옵니다. **오류가 아닙니다.**
  LangChain 이 외부 연동 모음 패키지를 2026년 5월부터 정리하고 있다는 안내입니다(Chroma 는 이미 `langchain-chroma` 로 분리).
- (선택) 실습 3 에서 작은 모델과 비교하려면: `ollama pull llama3.2:1b` (약 1.3GB)

## 실행 순서

```bash
# ── 1교시 (판서) 에서 읽는 코드 → 3교시에 직접 실행 ─────────────────────
python vector_stores.py              # FAISS = 검색 부품 / Chroma = 데이터베이스 · chroma_db/ 폴더 구경
python vector_stores.py --scores     # 같은 질문, 두 저장소의 거리 + 척도(l2 · cosine)
python vector_stores.py --filter     # ★★ chapter 필터 — 거른 뒤 찾기 vs 꺼낸 뒤 거르기
python rag_chain.py                  # 실습 1 — retriever | prompt | llm | parser
python rag_chain.py --types          # search_type 3종 (similarity · mmr · score_threshold)

# ── 2교시 (판서) 에서 읽는 코드 → 3교시에 직접 실행 ─────────────────────
python questions.py                          # 질문셋 보기 — 무료
python retriever_strategies.py --mmr         # λ 를 바꿔 가며
python retriever_strategies.py --multiquery  # ★★ 모델이 다시 쓴 질문을 본다
python retriever_strategies.py --hybrid      # 토큰 방식 · weights 를 바꿔 가며
python retriever_strategies.py               # ★★ 전체 비교표 ⏱ 먼저 걸어 두세요
python concepts_4.py                         # 개념 4종 + 7종 표 (무료)
python concepts_4.py --run compress          # (선택) Compression 체험
python concepts_4.py --run parent            # (선택) Parent Document 체험

# ── 3교시 실습 3 · 4 ──────────────────────────────────────────────
python lost_in_middle.py --repeat 3 --n 20                       # 실습 3
python lost_in_middle.py --repeat 3 --n 20 --model llama3.2:1b   # (선택) 작은 모델과 비교 ★
python citation.py                   # 실습 4 — 출처 표기 + 인용 번호 검증
python citation.py --pdf             # 같은 일을 학칙.pdf 인덱스로 — 쪽 번호까지
python citation.py --followup        # 후속 질문을 독립 질문으로 다시 쓰기
```

> ⏱ **`retriever_strategies.py`(전체)는 오래 걸립니다.** 질문 10개 × 전략 4종이고,
> MultiQuery 는 질문마다 LLM 호출이 하나 더 붙습니다. 실습 시작하자마자 먼저 걸어 두세요.

> 💡 `citation.py --pdf` 는 10주차 `slack_bot_mode/index_pdf/` 를 씁니다.
> 없으면 `10week_new/code/slack_bot_mode` 에서 `python build_index.py` 를 먼저 실행하세요 (학칙.pdf 필요).

## 실습 2 기록표 — 직접 채우세요 ★

| 질문 유형 | 기본 | MMR | MultiQuery | 하이브리드 | 무엇이 풀었나 |
|---|---|---|---|---|---|
| 표준 | | | | | |
| 구어체 | | | | | |
| 고유명사 (조문 번호) | | | | | |
| 넓은 질문 | | | | | |
| 없는 내용 | — | — | — | — | 답이 «찾을 수 없습니다» 였나 |

○ = 상위 3건에 정답 조가 있음 / × = 없음. 같은 유형의 질문이 여러 개면 «○ 개수 / 질문 수» 로 적습니다.

## 미니 프로젝트로 이어 가기 — 슬랙 봇에 전략 끼우기

10주차 `slack_bot_mode/rag_backend.py` 의 `_rag()` 안에서 **검색기를 만드는 한 줄**이 바꿔 끼울 자리입니다.

```python
# 지금 (기본 유사도 검색)
retriever = get_store().as_retriever(search_kwargs={"k": TOP_K})

# MMR 로 바꾼다면 — 한 줄
retriever = get_store().as_retriever(
    search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.7})
```

하이브리드는 `retriever_strategies.py` 의 `ko_tokens` · `bm25_docs` · `make_hybrid` 를 가져가 같은 자리에 넣습니다.
어느 전략이 내 프로젝트 질문에 맞는지는 **실습 2 기록표처럼 내 질문으로 재 보고** 고르세요.

## 과제 4 — 검색 전략 비교 + LangSmith 평가

- 검색 전략 3종 이상을 같은 질문셋에 적용 (실습 3종 중 2종 이상 + Compression · Parent Document · Self-Query 중 1종 이상)
- LangSmith 데이터셋(10~20개)으로 **정확성 · 근거성 · 검색 적합성** 3축 평가
- 결과표 + 어느 전략을 왜 고를지 한 문단
- ⚠️ 평가 결과는 **화면 캡처 필수** — 공유 링크는 보존기간(14일)이 지나면 열리지 않습니다
