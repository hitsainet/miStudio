"""Evidence that an old name is still used (042 FR-042.9; FTDD §4.1; TPD-2, TPD-3; CF-4, CF-5).

``legacy_name_uses`` holds one row per (kind, old name) ever used — never a value, never a row per
request or per day. ``src/core/legacy_names.record`` accepts only the pairs its own tables list, and the
POST route refuses any other pair, so the table holds at most ``len(ENV) + len(HOSTS) +
len(CATEGORY_ALIASES)`` rows (23 at P0). Rows are never deleted by the app; after P7 they are history.

``legacy_name_reporters`` holds one row per process name that ever reported (``api``, ``beat``, ``mcp``,
``worker:<celery node name>``): ``legacy_names_meta.reporters`` reads it. It grows by one row per new
reporter name (a worker's node name carries its pod hostname, so each rollout can add one row per
worker); nothing prunes it (an open question for the operator, recorded in the 042 P0 record).
"""

from sqlalchemy import CheckConstraint, Column, DateTime, String

from ..core.database import Base


class LegacyNameUse(Base):
    __tablename__ = "legacy_name_uses"
    __table_args__ = (CheckConstraint("kind IN ('env','host','category')", name="ck_legacy_name_uses_kind"),)

    kind = Column(String(16), primary_key=True)
    old = Column(String(255), primary_key=True)
    new = Column(String(255), nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_seen = Column(DateTime(timezone=True), nullable=False)


class LegacyNameReporter(Base):
    __tablename__ = "legacy_name_reporters"

    reporter = Column(String(64), primary_key=True)
    first_report = Column(DateTime(timezone=True), nullable=False)
    last_report = Column(DateTime(timezone=True), nullable=False)
