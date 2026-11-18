"""[2교시 코드 읽기 · 3교시 실습 4] 운영 관점의 LangSmith — 모니터링 · 피드백 수집 · 지속적 평가

  개발 중과 운영 중은 다릅니다

     |          | 개발 중 (6~7주차)          | 운영 중 (오늘)                    |
     | 누가 쓰나 | 개발자 본인                 | **실제 사용자**                   |
     | 입력     | 내가 만든 테스트 질문        | **예상 못 한 질문** ⚠️             |
     | 목적     | 버그 찾기 · 개선 측정        | **지금 잘 돌고 있나** · 만족하나    |
     | 보는 것  | Trace 하나를 뜯어봄          | **집계 지표**(오류율 · 지연 · 토큰) |
     | 평가     | 내가 만든 데이터셋           | **실제 로그에서 계속 보충**         |

  "배포하면 끝" 이 아니라 "배포하면 시작"

     개발 → 배포 → 관측 → 실패 수집 → 데이터셋 보충 → 개선 → 재평가 → …
                  └──────────────── 계속 돈다 ───────────────┘

  ⚠️ 운영 로그에는 **실제 사용자 입력**이 들어갑니다 (6주차 보안 원칙).

실행:
    python ops_feedback.py           # 개념 정리 (LLM 호출 없음)
    python ops_feedback.py --demo    # LLM 한 번 실행 → 그 run 에 👍 피드백 붙이기 (LangSmith 키 필요)
    python ops_feedback.py --stats   # 내 LangSmith 프로젝트의 최근 run 으로 오류율 · 지연 P50/P95 계산
"""

import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "gemma3:4b")
PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

CONCEPTS = """
── ① 모니터링 — 집계 지표 ─────────────────────────────────────

  | 지표              | 이상 신호                                              |
  | 오류율            | 갑자기 오르면 → 모델·API 장애, 입력 패턴 변화              |
  | 지연 분포 P50/P95 | 평균은 괜찮은데 **P95 가 튀면** 일부 사용자가 매우 느림 ⚠️  |
  | 토큰 · 비용        | 예상 밖 증가 → 순환이 과하게 돌고 있을 수 있음 (12~14주차)   |
  | 호출량            | 급증 → 무료 한도 소진 위험                                |

  ★ 평균이 아니라 분포를 봅니다.
    "평균 3초" 인데 P95 가 40초면 **20명 중 1명은 40초를 기다립니다.**

── ② 사용자 피드백 수집 ────────────────────────────────────────

  답변마다 run_id 를 기억해 두고, 사용자의 👍/👎 를 그 실행에 붙입니다.

      run_id = uuid.uuid4()
      chain.invoke(질문, {"run_id": run_id})                       # 실행에 내가 정한 id
      Client().create_feedback(run_id, key="user_rating", score=1)  # 👍 = 1, 👎 = 0

  | 왜 중요한가          | 설명                                                  |
  | 정답을 모르는 실운영  | 데이터셋의 기대 출력이 없다 — 사용자만 만족 여부를 안다    |
  | 데이터셋의 원천       | 👎 가 붙은 실행 → **다음 데이터셋 예제** (7주차)          |
  | 우선순위             | 어떤 유형에서 불만이 많은지 → 고칠 순서                   |

  슬랙 봇이라면: 답 메시지에 달린 👍/👎 이모지 반응(reaction_added 이벤트)을
  그 답의 run_id 에 붙이면 됩니다.

── ③ 지속적 평가 — 회귀 검사 ───────────────────────────────────

   프롬프트 수정 / 모델 교체 / 문서(학칙) 갱신
             ▼
   같은 데이터셋으로 다시 평가      ← 7주차 3교시 · 11주차 과제 4 그대로
             ▼
   점수가 떨어지지 않았으면 반영 / 떨어졌으면 되돌림

── 배포할 때 다시 보는 것 ───────────────────────────────────────

  | 항목          | 왜                               | 배운 곳                          |
  | 상태 저장소    | 재시작해도 대화가 남아야 함          | 13주차 체크포인터                 |
  | 비용 상한      | 순환 · 멀티 에이전트는 호출이 늘어남  | 12주차 recursion_limit · 14주차   |
  | 보안          | 도구 실행 권한 · 주입 방어            | 9주차 · 13주차 HITL               |
  | 관측          | 운영 중 무슨 일이 일어나는가          | 6~7주차 → 오늘                   |
  | 개인정보       | 추적에 질문 원문이 전송됨             | 6주차                            |
"""


def need_key() -> bool:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("⚠️ LANGSMITH_API_KEY 가 없습니다. 저장소 맨 위 .env 를 채우고 다시 실행하세요.")
        return True
    return False


def demo() -> None:
    """LLM 을 한 번 부르고, 그 실행(run)에 사용자 피드백을 붙인다."""
    if need_key():
        return
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
        print("⚠️ LANGSMITH_TRACING=true 가 아니면 run 이 기록되지 않아 피드백을 붙일 곳이 없습니다.")
        return

    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langchain_ollama import ChatOllama
    from langsmith import Client

    run_id = uuid.uuid4()  # ★ 실행에 붙일 id 를 내가 먼저 정한다
    q = "휴학은 한 번에 최대 얼마 동안 할 수 있나요? 한 문장으로."
    ans = ChatOllama(model=MODEL, temperature=0, num_predict=200).invoke(
        q, {"run_id": run_id, "run_name": "week14-feedback-demo"})
    print(f"Q: {q}\nA: {ans.content.strip()[:150]}")
    print(f"run_id: {run_id}  (프로젝트: {PROJECT})")

    wait_for_all_tracers()  # run 이 LangSmith 에 다 올라간 뒤에 피드백을 붙인다
    fb = Client().create_feedback(run_id, key="user_rating", score=1,
                                  comment="수업 실습 — 사용자가 👍 를 눌렀다고 가정")
    print(f"\n✅ 피드백을 붙였습니다 (feedback id {fb.id}).")
    print("   LangSmith → 프로젝트 → 이 run 을 열어 Feedback 칸에 user_rating 1 이 있는지 보세요.")


def stats(limit: int = 50) -> None:
    """최근 run 으로 집계 지표를 직접 계산한다 (LangSmith Monitor 탭이 보여 주는 것과 같은 종류)."""
    if need_key():
        return
    from langsmith import Client

    runs = [r for r in Client().list_runs(project_name=PROJECT, is_root=True, limit=limit)
            if r.end_time and r.start_time]
    if not runs:
        print(f"프로젝트 '{PROJECT}' 에 끝난 run 이 없습니다.")
        return

    lat = sorted((r.end_time - r.start_time).total_seconds() for r in runs)
    errors = sum(1 for r in runs if r.error)

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    print(f"프로젝트 '{PROJECT}' 최근 루트 run {len(runs)}개")
    print(f"  오류율   : {errors}/{len(runs)} = {errors / len(runs):.0%}")
    print(f"  지연 평균 : {sum(lat) / len(lat):6.1f}초")
    print(f"  지연 P50 : {pct(0.50):6.1f}초")
    print(f"  지연 P95 : {pct(0.95):6.1f}초   ← 평균과 얼마나 다른지 보세요")
    print(f"  가장 느림 : {lat[-1]:6.1f}초")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--demo" in sys.argv:
        demo()
    elif "--stats" in sys.argv:
        stats()
    else:
        print(CONCEPTS)


if __name__ == "__main__":
    main()
