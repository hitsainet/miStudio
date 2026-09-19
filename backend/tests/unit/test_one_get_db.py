"""There is exactly ONE ``get_db``, or an override cannot reach half the app.

WHY THIS FILE EXISTS. Until 2026-09-17 there were two ``get_db`` coroutines whose
bodies were byte-identical — ``core.database.get_db`` and ``core.deps.get_db`` —
built on the same ``AsyncSessionLocal``, with the same try/commit/rollback/close.
Identical behaviour, two distinct function OBJECTS.

FastAPI keys ``app.dependency_overrides`` on the object. ``conftest.py`` overrode
``core.deps.get_db``, which covered 17 endpoint modules and did NOTHING for the
68 ``Depends(get_db)`` sites in the six that import from ``core.database``:
saes (14), circuits (16), circuit_discovery (14), neuronpedia (10), steering (9)
and circuit_validation (5). Every request those endpoints served under test
opened a REAL session against ``settings.database_url``.

IT WAS NOT THEORETICAL. 17 test files drive those routes, and the dev database
``mistudio`` collected their fixtures: 20 model rows named ``test_model_*``,
``test_extraction_fail_*``, ``test_concurrent_model_*``, plus three
``external_saes`` rows.

WHY THE OBVIOUS GUARD IS ABSENT. Walking the app's routes and checking every
``get_db`` dependency CANNOT be written here. Under FastAPI 0.141 ``app.routes``
yields 10 entries of which only 4 are ``APIRoute``, none carrying a ``get_db``
dependency, while ``app.openapi()['paths']`` reports 250 real paths; recursing
into the ``_IncludedRouter`` finds 0 more. A guard built on that walk asserts
over an EMPTY set and passes forever — the fail-open shape this repo has hit
five times. So these tests enumerate modules the app actually loaded, and make
the denominator part of the assertion.

MUTATION CONTROLS (2026-09-17; applied alone, source restored by bytes and the
sha256 verified as e37d5969d55069c13ef0b116e453db0ac5be410953b21cb6ab3cdbe4203ca165):
  M1 restore the duplicate ``async def get_db`` in ``core/deps.py``
        -> test_the_two_modules_expose_the_same_object
        -> test_every_loaded_endpoint_module_shares_one_get_db

  M1 RUN 1 LEFT TWO SURVIVORS, and both were defects in THIS file:

  * ``test_the_six_formerly_escaping_modules_are_covered`` asserted that the six
    ``core.database`` importers satisfy ``is database.get_db``. That is
    tautologically true WHILE THE FORK EXISTS — they import from that very
    module. It named precisely the set that cannot detect the defect. Replaced
    by ``test_a_database_importer_and_a_deps_importer_share_one_object``, which
    compares ACROSS the two import paths.

  * ``test_an_override_reaches_a_formerly_escaping_endpoint`` committed a row and
    asserted the endpoint returned it. Under the fork the endpoint opens its own
    session against the SAME database, so a committed row comes back either way.
    Replaced by a flush-without-commit: a row inside an uncommitted transaction
    is visible only through the shared session, which discriminates no matter
    which database is configured.
"""

import sys

import pytest

# `import src.main` is here for its SIDE EFFECT: importing the app loads every
# endpoint module, so the `sys.modules` scan below is a record of what the
# application actually wired — not a scan of the source tree, which fails open.
import src.main  # noqa: F401
from src.core import database, deps

ENDPOINT_PREFIX = "src.api.v1.endpoints."

# Imported `get_db` from `core.database`, so an override keyed on `core.deps`
# never reached them. Asserting `is database.get_db` about THESE proves nothing.
FORMERLY_ESCAPING = (
    "saes",
    "circuits",
    "circuit_discovery",
    "circuit_validation",
    "neuronpedia",
    "steering",
)

# Imported from `core.deps`. Under the fork these resolve the DUPLICATE, so
# comparing one of these against one of the six above is what detects it.
VIA_DEPS = (
    "datasets",
    "models",
    "trainings",
    "features",
)

# 23 endpoint modules declared `Depends(get_db)` when this was written. The floor
# is lower so adding or removing one does not fail the suite, but high enough
# that an empty or near-empty discovery cannot pass.
MINIMUM_ENDPOINT_MODULES = 20


def _loaded_endpoint_modules():
    return {
        name[len(ENDPOINT_PREFIX):]: module
        for name, module in list(sys.modules.items())
        if name.startswith(ENDPOINT_PREFIX)
        and module is not None
        and hasattr(module, "get_db")
    }


class TestThereIsOnlyOneObject:
    def test_the_two_modules_expose_the_same_object(self):
        assert deps.get_db is database.get_db, (
            "core.deps.get_db must BE core.database.get_db, not a copy of it. "
            "Two identical-but-distinct objects are what let an override miss "
            "68 Depends(get_db) sites."
        )

    def test_every_loaded_endpoint_module_shares_one_get_db(self):
        modules = _loaded_endpoint_modules()

        # The denominator is part of the assertion: without this, a discovery
        # that returns nothing would satisfy the loop below vacuously.
        assert len(modules) >= MINIMUM_ENDPOINT_MODULES, (
            f"only {len(modules)} endpoint modules with a get_db were loaded; "
            f"expected at least {MINIMUM_ENDPOINT_MODULES}. The discovery is "
            "broken, so the identity check below would prove nothing."
        )

        divergent = sorted(
            name for name, module in modules.items()
            if module.get_db is not database.get_db
        )
        assert not divergent, (
            f"these modules resolve a DIFFERENT get_db object: {divergent}. "
            "An override keyed on the canonical object will not reach them."
        )

    def test_a_database_importer_and_a_deps_importer_share_one_object(self):
        """The cross-path check — the only identity check that detects a fork.

        M1 run 1 proved the naive version useless: the six `core.database`
        importers satisfy `is database.get_db` even while the duplicate exists.
        Comparing one of them against a `core.deps` importer does not.
        """
        modules = _loaded_endpoint_modules()

        for name in FORMERLY_ESCAPING + VIA_DEPS:
            assert name in modules, (
                f"{name} was not loaded, so this test did not check what it names."
            )

        escaping_name, deps_name = FORMERLY_ESCAPING[0], VIA_DEPS[0]
        assert modules[escaping_name].get_db is modules[deps_name].get_db, (
            f"{escaping_name} and {deps_name} resolve DIFFERENT get_db objects. "
            "An override keyed on one cannot reach the other, which is exactly "
            "how 68 Depends sites escaped the test session."
        )


class TestTheOverrideActuallyBites:
    """Identity is necessary and not sufficient.

    A refactor could satisfy every check above while reachability breaks again.
    This asks the only question that matters: is the endpoint running on the
    session the override supplied?
    """

    @pytest.mark.asyncio
    async def test_the_endpoint_shares_the_session_not_merely_the_database(
        self, client, async_session
    ):
        """Flush WITHOUT committing, so only the shared session can see the row.

        Committing instead cannot discriminate: under the fork the endpoint
        opens its own session against the same database and the row comes back
        regardless. M1 run 1 confirmed that version survived the mutation. A
        flushed-but-uncommitted row lives only inside this transaction.
        """
        from src.models.external_sae import ExternalSAE

        async_session.add(ExternalSAE(
            id="sae_one_get_db_guard", name="get_db guard row", source="local",
        ))
        await async_session.flush()  # deliberately NOT commit()

        response = await client.get("/api/v1/saes?limit=100")
        assert response.status_code == 200, response.text

        ids = [row["id"] for row in response.json()["data"]]
        assert "sae_one_get_db_guard" in ids, (
            "the saes endpoint did not see a row flushed in the test session, so "
            "it is running on a session of its own — the defect this file pins."
        )
