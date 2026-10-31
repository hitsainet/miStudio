"""runtime environment settings: overrides, append-only change history, ollama_url fold (Feature 036)

Revision ID: b36e0a7c1d42
Revises: a35d0e1f2c3b
Create Date: 2026-10-20

ENH-003 §3.1 / FTID 036 §3.2:

* ``environment_overrides`` — one row per operator override of an allow-listed setting
  (``core.config.RUNTIME_KEYS``). No row = the environment's value is in force.
* ``environment_setting_changes`` — append-only history (D12). A trigger refuses UPDATE and DELETE.
* D4: the ``app_settings`` row ``ollama_url`` (the only database-over-env setting before this
  feature) moves into ``environment_overrides``, with a history row saying so, and leaves
  ``app_settings``. An empty value is not carried: the old reader treated it as "use the
  environment", which is what no override means now.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "b36e0a7c1d42"
down_revision = "a35d0e1f2c3b"
branch_labels = None
depends_on = None

FOLD_REASON = "moved from app_settings (ENH-003 D4)"


def upgrade() -> None:
    op.create_table(
        "environment_overrides",
        sa.Column("key", sa.String(length=100), primary_key=True),
        sa.Column("value", JSONB, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_by", sa.String(length=100), server_default="operator", nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
    )
    op.create_table(
        "environment_setting_changes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("action", sa.String(length=10), nullable=False),
        sa.Column("previous_value", JSONB, nullable=True),
        sa.Column("previous_source", sa.String(length=10), nullable=False),
        sa.Column("new_value", JSONB, nullable=True),
        sa.Column("new_source", sa.String(length=10), nullable=False),
        sa.Column("actor", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(
        "ix_environment_setting_changes_changed_at", "environment_setting_changes", ["changed_at"]
    )
    op.create_index(
        "ix_environment_setting_changes_key_changed_at",
        "environment_setting_changes",
        ["key", "changed_at"],
    )
    op.execute(
        """
CREATE OR REPLACE FUNCTION environment_setting_changes_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'environment_setting_changes is append-only (% refused)', TG_OP;
END;
$$ LANGUAGE plpgsql
"""
    )
    op.execute(
        """
CREATE TRIGGER trg_environment_setting_changes_append_only
BEFORE UPDATE OR DELETE ON environment_setting_changes
FOR EACH ROW EXECUTE FUNCTION environment_setting_changes_append_only()
"""
    )

    # D4 fold. `app_settings` may not exist on a scratch database that only replays this file.
    conn = op.get_bind()
    if sa.inspect(conn).has_table("app_settings"):
        row = conn.execute(
            sa.text(
                "SELECT value FROM app_settings WHERE key = 'ollama_url' AND is_sensitive = false"
            )
        ).first()
        value = (row[0] or "").strip() if row is not None else ""
        if value:
            conn.execute(
                sa.text(
                    "INSERT INTO environment_overrides (key, value, updated_by, reason) "
                    "VALUES ('ollama_url', to_jsonb(CAST(:v AS text)), 'migration', :r)"
                ),
                {"v": value, "r": FOLD_REASON},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO environment_setting_changes "
                    "(key, action, previous_value, previous_source, new_value, new_source, actor, reason) "
                    "VALUES ('ollama_url', 'set', NULL, 'env', to_jsonb(CAST(:v AS text)), 'override', "
                    "'migration', :r)"
                ),
                {"v": value, "r": FOLD_REASON},
            )
        if row is not None:
            conn.execute(sa.text("DELETE FROM app_settings WHERE key = 'ollama_url'"))


def downgrade() -> None:
    conn = op.get_bind()
    if sa.inspect(conn).has_table("app_settings"):
        row = conn.execute(
            sa.text(
                "SELECT value #>> '{}' FROM environment_overrides "
                "WHERE key = 'ollama_url' AND jsonb_typeof(value) = 'string'"
            )
        ).first()
        if row is not None and row[0]:
            conn.execute(
                sa.text(
                    "INSERT INTO app_settings (id, key, value, is_sensitive, category) "
                    "VALUES (gen_random_uuid(), 'ollama_url', :v, false, 'endpoints') "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ),
                {"v": row[0]},
            )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_environment_setting_changes_append_only "
        "ON environment_setting_changes"
    )
    op.execute("DROP FUNCTION IF EXISTS environment_setting_changes_append_only()")
    op.drop_index("ix_environment_setting_changes_key_changed_at", table_name="environment_setting_changes")
    op.drop_index("ix_environment_setting_changes_changed_at", table_name="environment_setting_changes")
    op.drop_table("environment_setting_changes")
    op.drop_table("environment_overrides")
