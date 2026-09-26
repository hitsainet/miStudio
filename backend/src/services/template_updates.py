"""Refusing a PATCH that would write NULL into a column that cannot hold one (review R2F-10).

Every template service applied ``updates.model_dump(exclude_unset=True)`` straight
onto the ORM row. ``exclude_unset`` deliberately KEEPS a field the client sent
explicitly as null — that is how "clear this description" is expressed — so
``PATCH {"name": null}`` reached the database as ``SET name = NULL`` against a
``nullable=False`` column. The IntegrityError escaped the endpoint as a 500 on a
request that is merely invalid, and it did so in all four template services.

The required columns are derived from the MODEL rather than from a hand-kept list
per service. A list stops covering a column added later, silently, which is the
failure mode this repo keeps rediscovering; the mapper cannot drift from itself.

Only keys the request actually sent are examined, so nothing else about the PATCH
semantics changes: an omitted field is still untouched, and a nullable column can
still be cleared.
"""

from typing import Any, Dict, FrozenSet, Mapping

from sqlalchemy import inspect as sa_inspect


class NullNotAllowed(ValueError):
    """A PATCH sent null for a column that cannot hold null.

    Derives from ``ValueError`` so the endpoints' existing ``except ValueError``
    handling maps it to a 4xx instead of letting it surface as a 500.
    """


def required_columns(model_class: Any) -> FrozenSet[str]:
    """Attribute names of the model's NOT NULL columns, primary keys aside.

    A Python-side ``default=`` does not rescue an explicit null: SQLAlchemy applies
    a default only when the attribute was never set, so ``is_favorite = None`` on an
    existing row emits ``SET is_favorite = NULL`` and violates the constraint. The
    only thing that matters here is ``nullable``.
    """
    return frozenset(
        key
        for key, column in sa_inspect(model_class).columns.items()
        if not column.nullable and not column.primary_key
    )


def reject_null_updates(update_data: Mapping[str, Any], model_class: Any) -> Dict[str, Any]:
    """``update_data`` unchanged, or raise :class:`NullNotAllowed` naming every bad field."""
    offending = sorted(
        field
        for field, value in update_data.items()
        if value is None and field in required_columns(model_class)
    )
    if offending:
        raise NullNotAllowed(
            f"{', '.join(offending)} cannot be null. Send a value, or omit the field "
            "to leave it unchanged."
        )
    return dict(update_data)
