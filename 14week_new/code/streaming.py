"""[2교시 코드 읽기 · 3교시 실습 2] 에이전트의 진행 상황을 흘려보낸다 — 상태 단위 / 토큰 단위

  에이전트는 오래 걸립니다.

     [4주차 단일 호출]   질문 ─── 몇 초 ───▶ 답변
     [14주차 멀티 에이전트] 질문 ─── 수십 초 ─────────────────▶ 답변  ⚠️
                              그동안 화면에 아무것도 없으면 → 사용자는 멈춘 줄 압니다

  두 종류의 스트리밍

     | 종류         | 무엇이 흐르나         | 사용자가 보는 것                      |
     | 토큰 단위     | 글자                 | 답이 한 글자씩 흘러나옴 (4주차 stream)  |
     | **상태 단위** | **노드 하나가 끝날 때** | "학칙을 찾는 중… 계산하는 중… 정리 중…" |

     토큰 단위만으로는 **도구를 실행하는 동안 여전히 조용합니다** (글을 만드는 중이 아니므로).
     "지금 무엇을 하고 있는가" 를 보여 주는 것이 상태 단위입니다.

     | stream_mode | 나오는 것                       | 쓰임         |
     | "values"    | 노드가 끝날 때마다 **전체 상태**  | 디버깅 (13주차) |
     | "updates"   | **어느 노드가 무엇을 바꿨는지**   | 진행 표시      |
     | "messages"  | LLM 의 **토큰** + 어느 노드인지   | 답 글자 흘리기  |

  📌 총 시간은 줄지 않습니다. **사용자가 기다릴 수 있게** 만드는 것이 목표입니다.
     (4주차 결론 그대로 — 스트리밍은 속도 최적화가 아니라 사용자 경험 개선)

실행:
    python streaming.py              # 상태 단위 → 토큰 단위 (그래프를 두 번 돌립니다)
    python streaming.py --updates    # 상태 단위만
    python streaming.py --subgraphs  # 상태 단위 + 서브그래프 안쪽 노드까지
    python streaming.py --tokens     # 토큰 단위만 (writer 노드로 한정)
"""

import sys
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessageChunk

import supervisor
from supervisor import CONFIG, INPUTS, QUESTION, graph  # 1교시 그래프를 그대로 씁니다

supervisor.VERBOSE = False  # 노드 안의 print 는 끄고, 진행은 이 파일이 stream 으로 보여 준다

# 노드 이름을 **사용자 말로 바꿉니다** — "researcher" 가 아니라 "학칙을 찾는 중"
LABELS = {
    "supervisor": "담당자를 정하는 중",
    "researcher": "학칙을 찾는 중",
    "calculator": "계산하는 중",
    "writer": "답변을 정리하는 중",
    "retrieve": "  └ 학칙을 검색하는 중",       # researcher 서브그래프 안쪽
    "summarize": "  └ 찾은 조문을 요약하는 중",
    "agent": "  └ 모델이 생각하는 중",         # calculator 서브그래프 안쪽
    "tools": "  └ 도구를 실행하는 중",
}


def stream_updates(subgraphs: bool = False) -> float:
    """① 상태 단위 — 노드 하나가 끝날 때마다 한 줄"""
    print("── 상태 단위 (stream_mode='updates') ─────────────")
    t0 = time.perf_counter()
    for item in graph.stream(INPUTS, CONFIG, stream_mode="updates", subgraphs=subgraphs):
        ns, chunk = item if subgraphs else ((), item)
        for node, update in chunk.items():
            where = f" ({ns[0].split(':')[0]} 안)" if ns else ""
            extra = f" → {update['next']}" if node == "supervisor" and update else ""
            print(f"  ▸ [{time.perf_counter() - t0:5.1f}s] {LABELS.get(node, node)}{where} 끝{extra}")
    total = time.perf_counter() - t0
    print(f"  ▸ [{total:5.1f}s] 완료")
    return total


def stream_tokens() -> None:
    """② 토큰 단위 — writer 노드의 토큰만 흘린다 (중간 노드까지 흘리면 화면이 지저분해짐)"""
    print("\n── 토큰 단위 (stream_mode='messages') ────────────")
    print("--- 최종 답변 ---")
    t0 = time.perf_counter()
    first = None
    for msg, meta in graph.stream(INPUTS, CONFIG, stream_mode="messages"):
        # AIMessageChunk = 생성 중인 토큰 조각. (노드가 상태에 넣은 완성 메시지도 한 번 더 오므로 거른다)
        if meta.get("langgraph_node") == "writer" and isinstance(msg, AIMessageChunk) and msg.content:
            first = first or time.perf_counter() - t0
            print(msg.content, end="", flush=True)
    print(f"\n(첫 글자까지 {first or 0:.1f}s / 전체 {time.perf_counter() - t0:.1f}s)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Q: {QUESTION}\n")

    if "--tokens" in sys.argv:
        stream_tokens()
        return

    stream_updates(subgraphs="--subgraphs" in sys.argv)

    if "--updates" not in sys.argv and "--subgraphs" not in sys.argv:
        stream_tokens()


if __name__ == "__main__":
    main()
