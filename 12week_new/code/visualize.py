"""[3교시 / 실습 4] 그래프 시각화 + 실행 경로 확인

  그래프는 compile() 한 뒤 get_graph() 로 '모양' 을 꺼낼 수 있습니다. 그리는 방법은 세 가지입니다.

  | 방법              | 필요한 것                     | 순환(되돌아가는 화살표)       |
  | draw_ascii()      | grandalf 패키지               | ⚠️ 그리지 못한다 — 아래로만 이어진다 |
  | draw_mermaid()    | 추가 설치 없음                | ✅ "evaluate -. revise .-> generate" 한 줄로 나온다 |
  | draw_mermaid_png()| 인터넷 (mermaid.ink 서비스)    | ✅ 그림으로                      |

  ★ ASCII 그림만 보고 "순환이 없네?" 라고 판단하지 마세요. Mermaid 텍스트에서 화살표를 확인합니다.
    Mermaid 텍스트는 GitHub README 에 ```mermaid 블록으로 붙이면 그림으로 보이고,
    https://mermaid.live 에 붙여 넣어도 그림이 됩니다.

  ★ 과제 5 의 '시각화' 는 ASCII 캡처 · Mermaid 텍스트 · PNG 중 무엇이든 됩니다.

실행:
    python visualize.py                    # reflection 그래프 (순환)
    python visualize.py branch             # branch_graph 그래프 (세 갈래 분기)
    python visualize.py router             # rag_router_graph 그래프 (학칙 / 일반 분기)
    python visualize.py first              # first_graph 그래프 (직렬)
    python visualize.py my_bot_graph       # ★ 내가 만든 그래프 파일 (같은 폴더의 .py, 안에 graph 가 있어야 함)
    python visualize.py reflection --png   # PNG 저장 시도 (인터넷 필요)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent

GRAPHS = {
    "reflection": ("reflection", "순환 — Mermaid 에서 되돌아가는 화살표(revise)를 찾아보세요"),
    "branch": ("branch_graph", "분기 — 세 갈래로 나뉘는 것을 확인하세요"),
    "router": ("rag_router_graph", "분기 — 학칙 질문만 검색으로 갑니다 (미니 프로젝트 골격)"),
    "first": ("first_graph", "직렬 — 분기도 순환도 없습니다 (이 정도면 LCEL 이 더 짧습니다)"),
}


def load_graph(key: str):
    import importlib

    module_name = GRAPHS[key][0] if key in GRAPHS else key  # 표에 없으면 그 이름의 .py 파일
    return importlib.import_module(module_name).graph  # 모듈을 읽기만 합니다 — LLM 호출 없음


def draw_ascii(graph) -> bool:
    """ASCII — grandalf 가 필요합니다."""
    try:
        print(graph.get_graph().draw_ascii())
        return True
    except ImportError:
        print("""⚠️ draw_ascii() 에는 grandalf 가 필요합니다.
     pip install grandalf
   (설치가 안 되면 아래 Mermaid 로 충분합니다)
""")
        return False


def draw_mermaid(graph) -> str:
    """Mermaid 텍스트 — 추가 설치 없이 됩니다. README 에 붙이기 좋습니다 ★"""
    text = graph.get_graph().draw_mermaid()
    print(text)
    return text


def draw_png(graph, out: Path) -> None:
    """PNG — 기본 방식은 mermaid.ink 웹 서비스에 그림을 부탁합니다 (인터넷 필요)."""
    try:
        out.write_bytes(graph.get_graph().draw_mermaid_png())
        print(f"✅ 저장: {out}")
    except Exception as e:
        print(f"""⚠️ PNG 렌더링 실패 ({type(e).__name__}: {str(e)[:80]})
   인터넷 연결이 필요합니다. ASCII / Mermaid 로 충분합니다.
""")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    names = [a.removesuffix(".py") for a in sys.argv[1:] if not a.startswith("--")]
    key = names[0] if names else "reflection"
    if key not in GRAPHS and not (HERE / f"{key}.py").exists():
        sys.exit(f"⚠️ {key}.py 가 없습니다. 고를 수 있는 것: {' · '.join(GRAPHS)} · 또는 같은 폴더의 내 그래프 파일 이름")
    note = GRAPHS[key][1] if key in GRAPHS else "내 그래프 — 갈래마다 점선 화살표가 있는지 확인하세요"

    print(f"── {key} 그래프 — {note}\n")
    graph = load_graph(key)

    print("── ① ASCII ─────────────────────────────────────")
    draw_ascii(graph)

    print("\n── ② Mermaid (README 에 그대로 붙이세요) ────────")
    text = draw_mermaid(graph)

    mmd = HERE / f"{key}.mmd"
    mmd.write_text(text, encoding="utf-8")
    print(f"\n✅ 저장: {mmd}")

    if "--png" in sys.argv:
        print("\n── ③ PNG ────────────────────────────────────")
        draw_png(graph, HERE / f"{key}.png")

    print("""
── Mermaid 읽는 법 ───────────────────────────────────────
   A --> B            보통 엣지 (항상 간다)
   A -. 이름 .-> B    조건부 엣지 (라우팅 함수가 '이름' 을 돌려줄 때만 간다)
   A -.-> B           조건부 엣지인데 이름이 없다 = 돌려주는 이름이 노드 이름(B)과 같아서 생략된 것

── LangSmith 에서 실행 경로 보기 ★ ────────────────────────
   python reflection.py 를 추적을 켠 채 실행한 뒤 LangSmith 에서 그 실행을 열면:

   LangGraph
    ├ generate    (1회차)
    ├ evaluate    (1회차)
    ├ generate    (2회차)   ★ 같은 노드가 다시 나온다 = 순환이 실제로 돌았다는 증거
    ├ evaluate    (2회차)
    └ ...

  | 확인할 것              | 왜                                |
  | 같은 노드가 여러 번     | 순환이 실제로 돌았다는 증거           |
  | 회차별 지적 사항        | 무엇을 지적했고 반영됐는지            |
  | 총 지연 · 토큰          | 반복의 비용을 숫자로 (6주차)          |
  | 어느 분기를 탔는가      | 조건부 엣지의 판단 결과               |
""")


if __name__ == "__main__":
    main()
