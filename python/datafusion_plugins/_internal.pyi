from typing import Any

__version__: str

class AsofJoinTable:
    def __init__(
        self,
        left: bytes,
        right: bytes,
        left_on: str,
        right_on: str,
        by_left: list[str],
        by_right: list[str],
        allow_exact_matches: bool = True,
        suffix: str = "_right",
        check_sortedness: bool = True,
    ) -> None: ...
    def __datafusion_table_provider__(self, session: Any) -> object: ...

def last_plan() -> str | None: ...
def ffi_version() -> int: ...
