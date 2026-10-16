from typing import Any

__version__: str

class PlanRouteUnavailable(ValueError): ...

class NodeTable:
    @staticmethod
    def from_plans(node: str, inputs: list[bytes], args: dict[str, Any]) -> NodeTable: ...
    @staticmethod
    def from_frames(node: str, inputs: list[Any], args: dict[str, Any]) -> NodeTable: ...
    def last_plan(self) -> str | None: ...
    def __datafusion_table_provider__(self, session: Any) -> object: ...

def set_memory_limit(limit: int | None, pool: str = "fair") -> None:
    """Limit the memory of node queries run from now on (None: unbounded).

    A limit set on the datafusion SessionContext does not reach the plugin:
    DataFusion 55's FFI passes the plugin a default, unbounded pool.
    """
