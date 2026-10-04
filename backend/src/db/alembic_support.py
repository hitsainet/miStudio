"""What Alembic compares the database against — defined once.

``alembic/env.py`` and ``tests/unit/test_schema_guards.py`` both import this module,
so the metadata and options behind ``alembic check`` are the ones the schema guards
test. Before this existed, env.py registered models by importing one class
(``from src.models import Dataset``), which silently left ``task_queue`` out of every
autogenerate comparison.
"""

from sqlalchemy import MetaData

# Types and server defaults are part of the schema. Without these, autogenerate and
# `alembic check` report a matching schema over columns whose type or default differs.
COMPARE_OPTIONS = {"compare_type": True, "compare_server_default": True}


def target_metadata() -> MetaData:
    """``Base.metadata`` with every model registered.

    The import is inside the function so a fresh interpreter (which is what
    ``alembic`` is) registers the models no matter what else it has imported.
    """
    import src.models  # noqa: F401  registers every model on Base.metadata
    from src.core.database import Base

    return Base.metadata
