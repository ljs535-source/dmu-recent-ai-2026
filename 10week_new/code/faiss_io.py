"""[공용] FAISS 인덱스 저장·읽기 — ⚠️ 한글 경로 대비책 ★★

  ⚠️⚠️ 실습실에서 실제로 터지는 문제입니다.

     FAISS 의 save_local() / load_local() 은 내부적으로 C++ 함수
     (faiss.write_index / faiss.read_index) 를 호출합니다.
     이 함수들은 **경로에 ASCII 가 아닌 문자(한글 등)가 있으면 실패합니다.**

         RuntimeError: Error in FileIOWriter... could not open
             ...\\최신인공지능\\...\\index.faiss for writing: Illegal byte sequence

     저장소가 `C:\\Users\\홍길동\\바탕화면\\수업\\...` 같은 경로에 있으면
     **10주차 마지막 단계에서 그대로 멈춥니다.** 그리고 11·13주차가 연쇄로 막힙니다. ⚠️

  대응 (이 파일이 하는 일)

     ① 먼저 표준 API 를 그대로 씁니다      save_local / load_local
        → 경로가 ASCII 면 슬라이드 코드와 똑같이 동작합니다. ★
     ② 실패하면 바이트로 직렬화해 저장합니다  serialize_to_bytes / deserialize_from_bytes
        → 파이썬의 open() 이 쓰므로 한글 경로에서도 됩니다.

  💡 **가장 깔끔한 해결은 저장소를 영문 경로에 두는 것입니다.**
     예: C:\\dev\\langchain-2026\\

이 파일은 단독 실행용이 아닙니다. embed.py 등이 가져다 씁니다.
(단독으로 돌리면 현재 경로가 안전한지만 알려줍니다 — 수업 도입부에 쓰십시오 ★)
"""

from __future__ import annotations

from pathlib import Path

BYTES_NAME = "index.bytes"  # ② 대체 저장 형식


def path_is_ascii(path: str | Path) -> bool:
    """이 경로에서 FAISS 의 C++ 저장이 될 것인가."""
    return str(Path(path).resolve()).isascii()


def save_index(store, directory: str | Path) -> str:
    """인덱스를 저장하고, 어떤 방식으로 저장했는지 돌려준다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    try:
        store.save_local(str(directory))  # ① 표준 API — 슬라이드 코드 그대로 ★
        return "save_local"
    except Exception as e:  # ⚠️ 한글 경로 등
        (directory / BYTES_NAME).write_bytes(store.serialize_to_bytes())  # ②
        print(f"""  ⚠️ save_local() 이 실패해 바이트 방식으로 저장했습니다.
     원인: {type(e).__name__} — 경로에 한글이 있으면 FAISS 의 C++ 저장이 실패합니다. ★
     읽기도 이 파일(faiss_io.py)을 통해서만 됩니다.
     💡 근본 해결: 저장소를 **영문 경로**로 옮기십시오 (예: C:\\dev\\langchain-2026).""")
        return "bytes"


def load_index(directory: str | Path, embeddings):
    """저장 방식을 자동으로 판별해 읽는다. 없으면 None."""
    from langchain_community.vectorstores import FAISS

    directory = Path(directory)

    if (directory / "index.faiss").exists():
        # allow_dangerous_deserialization: 내가 만든 로컬 파일이므로 신뢰합니다.
        # ⚠️ 남이 준 인덱스 파일에는 켜지 마십시오 — pickle 을 읽습니다.
        return FAISS.load_local(str(directory), embeddings,
                                allow_dangerous_deserialization=True)

    if (directory / BYTES_NAME).exists():
        return FAISS.deserialize_from_bytes(
            embeddings=embeddings,
            serialized=(directory / BYTES_NAME).read_bytes(),
            allow_dangerous_deserialization=True,
        )

    return None


def index_exists(directory: str | Path) -> bool:
    directory = Path(directory)
    return (directory / "index.faiss").exists() or (directory / BYTES_NAME).exists()


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    here = Path(__file__).parent
    ok = path_is_ascii(here)
    print(f"현재 경로: {here}")
    print(f"경로가 ASCII 인가: {'✅ 예 — 표준 save_local 로 저장됩니다' if ok else '❌ 아니오'}")
    if not ok:
        print("""
  ⚠️ 경로에 한글(또는 비 ASCII 문자)이 있습니다.
     FAISS 의 save_local() / load_local() 이 실패합니다.
     이 파일이 바이트 방식으로 우회하지만, **근본 해결은 영문 경로**입니다. ★

         예:  C:\\dev\\langchain-2026\\week10\\

""")
