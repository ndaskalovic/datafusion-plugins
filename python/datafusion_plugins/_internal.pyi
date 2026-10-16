from typing import Any

__version__: str

class PlanTable:
    def __init__(self, node: str, inputs: list[bytes], args: dict[str, Any]) -> None: ...
    def __datafusion_table_provider__(self, session: Any) -> object: ...

class StreamTable:
    def __init__(self, node: str, inputs: list[Any], args: dict[str, Any]) -> None: ...
    def __datafusion_table_provider__(self, session: Any) -> object: ...

def last_plan() -> str | None: ...
def ffi_version() -> int: ...
def set_memory_limit(limit: int | None, pool: str = "fair") -> None:
    """Limit the memory of the plugin's plans (None: unbounded).

    A limit set on the datafusion SessionContext does not reach the plugin:
    DataFusion 55's FFI passes the plugin a default, unbounded pool.
    """
