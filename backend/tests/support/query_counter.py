"""Count the SQL statements a block executes (Feature 040 FR-6; FTID §11).

`count_queries(engine)` is a context manager over a sync `Engine` or an `AsyncEngine` (its
`sync_engine` is where SQLAlchemy fires `before_cursor_execute`). It counts every statement the
engine sends, so a per-row query loop shows up as a count that grows with the rows.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List

from sqlalchemy import event


class QueryCount:
    def __init__(self) -> None:
        self.statements: List[str] = []

    @property
    def count(self) -> int:
        return len(self.statements)


@contextmanager
def count_queries(engine) -> Iterator[QueryCount]:
    target = getattr(engine, "sync_engine", engine)
    counter = QueryCount()

    def _before(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        counter.statements.append(statement)

    event.listen(target, "before_cursor_execute", _before)
    try:
        yield counter
    finally:
        event.remove(target, "before_cursor_execute", _before)
