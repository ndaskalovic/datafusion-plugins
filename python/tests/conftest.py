"""Shared fixtures."""

from __future__ import annotations

import pytest
from datafusion import SessionConfig, SessionContext


def make_ctx() -> SessionContext:
    config = (
        SessionConfig()
        .with_target_partitions(4)
        .with_batch_size(64)
        # Tiny files still split into many byte ranges...
        .set("datafusion.optimizer.repartition_file_min_size", "1")
        # ...each read in order (no stealing), so a sorted file's partitions
        # are each sorted for the host route too.
        .set("datafusion.execution.enable_file_stream_work_stealing", "false")
    )
    return SessionContext(config)


@pytest.fixture
def ctx() -> SessionContext:
    return make_ctx()
