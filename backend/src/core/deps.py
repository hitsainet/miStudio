"""
FastAPI dependencies.

This module provides dependency injection functions for FastAPI routes.

``get_db`` IS RE-EXPORTED HERE, NOT REDEFINED — and that is load-bearing.

Until 2026-09-17 this module defined its own ``get_db`` whose body was
byte-identical to ``core.database.get_db``: the same ``AsyncSessionLocal``, the
same try/commit/rollback/close. Two functions, identical behaviour, and — the
part that mattered — two distinct function OBJECTS.

FastAPI keys ``app.dependency_overrides`` on the function object. ``conftest.py``
overrode ``core.deps.get_db``, so it did nothing at all for the 68
``Depends(get_db)`` sites in the six modules that import from ``core.database``:
``saes``, ``circuits``, ``circuit_discovery``, ``circuit_validation``,
``neuronpedia`` and ``steering``. Every request those endpoints served in a test
opened a REAL session against ``settings.database_url`` instead of the test
session.

That is not theoretical. 17 test files exercise those routes, and the dev
database ``mistudio`` accumulated their fixtures — 20 model rows named
``test_model_*``, ``test_extraction_fail_*``, ``test_concurrent_model_*``, plus
three ``external_saes`` rows. When that residue no longer contained ``m_dl``,
``test_sae_hook_recorded_at_import`` began failing with a foreign key violation
against a row its own fixture had written — to a different database.

This is the shape this repo keeps finding: a guard satisfied by the wrong
occurrence. The fix is not to override both names; it is to leave exactly one
object for every caller to reach.

``tests/unit/test_one_get_db.py`` pins it, and asserts the WIRING — that an
override reaches an endpoint in a formerly-escaping module — not merely that the
two names are equal.
"""

from .database import get_db

__all__ = ["get_db"]
