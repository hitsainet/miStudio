"""Runtime environment overrides and their change history (Feature 036; ENH-003 D4, D12).

THE ENVIRONMENT IS THE SOURCE OF TRUTH. A row in ``environment_overrides`` replaces the environment's
value of ONE allow-listed key (``core.config.RUNTIME_KEYS``) until the operator clears it; clearing
deletes the row. Readers never touch this table: they call ``core.runtime_settings.effective()``,
whose cache is filled from it at three wiring points (after a write, at API startup, at every Celery
``task_prerun``).

``environment_setting_changes`` is APPEND-ONLY. Every set and every reset adds one row; nothing
updates or deletes one. That is enforced twice: here, by mapper events that raise on an ORM UPDATE
or DELETE, and in the migration, by a trigger that raises for raw SQL. Secrets never reach it,
because no secret is in ``RUNTIME_KEYS`` (``test_settings_readers.py`` pins the disjointness).
"""

from sqlalchemy import BigInteger, Column, DateTime, Index, String, Text, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class EnvironmentOverride(Base):
    """One operator override of one runtime-writable setting."""

    __tablename__ = "environment_overrides"

    key = Column(String(100), primary_key=True)
    # JSON-typed so a boolean stays a boolean; a JSON null is a real override (it clears an
    # optional URL), distinct from "no row" (use the environment's value).
    value = Column(JSONB, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_by = Column(String(100), nullable=False, server_default="operator")
    reason = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<EnvironmentOverride(key={self.key})>"


class EnvironmentSettingChange(Base):
    """One set or reset of a runtime setting. Append-only."""

    __tablename__ = "environment_setting_changes"
    __table_args__ = (
        Index("ix_environment_setting_changes_changed_at", "changed_at"),
        Index("ix_environment_setting_changes_key_changed_at", "key", "changed_at"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False)
    action = Column(String(10), nullable=False)  # 'set' | 'reset'
    previous_value = Column(JSONB, nullable=True)
    previous_source = Column(String(10), nullable=False)  # 'env' | 'override'
    new_value = Column(JSONB, nullable=True)
    new_source = Column(String(10), nullable=False)  # 'env' | 'override'
    actor = Column(String(100), nullable=False)
    reason = Column(Text, nullable=True)
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<EnvironmentSettingChange(key={self.key}, action={self.action})>"


class HistoryIsAppendOnly(RuntimeError):
    """Raised when code tries to change or remove a history row."""


@event.listens_for(EnvironmentSettingChange, "before_update")
def _refuse_update(mapper, connection, target):  # noqa: ARG001
    raise HistoryIsAppendOnly("environment_setting_changes is append-only: a row cannot be updated")


@event.listens_for(EnvironmentSettingChange, "before_delete")
def _refuse_delete(mapper, connection, target):  # noqa: ARG001
    raise HistoryIsAppendOnly("environment_setting_changes is append-only: a row cannot be deleted")


#: The database-level half of "append-only": raw SQL cannot change or remove a history row either.
#: Installed by the migration and, through this DDL hook, by ``create_all`` (the unit suite's schema),
#: so both schemas carry it. TRUNCATE is deliberately not blocked (a deliberate wipe, test teardown).
APPEND_ONLY_FUNCTION = "environment_setting_changes_append_only"
APPEND_ONLY_TRIGGER = "trg_environment_setting_changes_append_only"
APPEND_ONLY_DDL = (
    f"""
CREATE OR REPLACE FUNCTION {APPEND_ONLY_FUNCTION}() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'environment_setting_changes is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
""",
    f"""
CREATE TRIGGER {APPEND_ONLY_TRIGGER}
BEFORE UPDATE OR DELETE ON environment_setting_changes
FOR EACH ROW EXECUTE FUNCTION {APPEND_ONLY_FUNCTION}()
""",
)


def _install_append_only_trigger(target, connection, **kw):  # noqa: ARG001
    if connection.dialect.name != "postgresql":
        return
    from sqlalchemy import text

    for statement in APPEND_ONLY_DDL:
        connection.execute(text(statement))


event.listen(EnvironmentSettingChange.__table__, "after_create", _install_append_only_trigger)
