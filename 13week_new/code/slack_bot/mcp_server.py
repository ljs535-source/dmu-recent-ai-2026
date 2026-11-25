"""[13주차 · 미니 프로젝트 ③] MCP 서버 — 봇이 쓰는 «메모» 도구 세 개 (FastMCP)

   9주차 mcp_test/mcp_server.py 와 같은 모양입니다.  @mcp.tool 을 붙이면 MCP 도구가 됩니다.

       도구           하는 일                       승인
       list_notes     메모 목록 보기 (읽기)          필요 없음
       add_note       메모 하나 추가 (쓰기)          ⏸ 필요 — bot_graph.DANGEROUS
       delete_note    메모 하나 삭제 (되돌릴 수 없음)  ⏸ 필요 — bot_graph.DANGEROUS

   ★ 승인(interrupt)은 이 서버가 아니라 봇 그래프(bot_graph.py 의 tools 노드)에서 겁니다.
     MCP 서버는 «무엇을 할 수 있나» 만 내놓고, «해도 되나» 는 봇이 사람에게 묻습니다 (9주차 방어 ④).

   메모는 이 폴더의 notes.json 에 저장됩니다 (없으면 처음 쓸 때 만들어집니다).
   미니 프로젝트에서는 이 세 도구를 내 프로젝트 도구(일정 조회 · 공지 발송 · 파일 저장 …)로 바꾸면 됩니다.

  ⚠️ 이 파일은 직접 실행하지 않습니다. bot_graph.py 가 자식 프로세스로 띄우고, 끝나면 종료합니다.
  ⚠️ 이 파일에서는 print() 를 쓰지 않습니다 — stdio 전송은 표준출력이 곧 통신 통로입니다 (9주차).
"""

import json
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

NOTES = Path(__file__).with_name("notes.json")

mcp = FastMCP("week13-notes")


def _load() -> list[dict]:
    if not NOTES.exists():
        return []
    return json.loads(NOTES.read_text(encoding="utf-8"))


def _save(notes: list[dict]) -> None:
    NOTES.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")


@mcp.tool
def list_notes() -> str:
    """저장된 메모 목록을 번호와 함께 보여 준다."""
    notes = _load()
    if not notes:
        return "저장된 메모가 없습니다."
    return "\n".join(f"{n['id']}. {n['text']}  ({n['at']})" for n in notes)


@mcp.tool
def add_note(text: str) -> str:
    """메모 하나를 저장한다. text 는 저장할 메모 내용이다."""
    notes = _load()
    new_id = max((n["id"] for n in notes), default=0) + 1
    notes.append({"id": new_id, "text": text, "at": datetime.now().strftime("%m-%d %H:%M")})
    _save(notes)
    return f"{new_id}번 메모를 저장했습니다: {text}"


@mcp.tool
def delete_note(note_id: int) -> str:
    """번호(note_id)로 메모 하나를 삭제한다. 되돌릴 수 없다."""
    notes = _load()
    left = [n for n in notes if n["id"] != note_id]
    if len(left) == len(notes):
        return f"{note_id}번 메모가 없습니다."
    _save(left)
    return f"{note_id}번 메모를 삭제했습니다."


if __name__ == "__main__":
    mcp.run(show_banner=False)
