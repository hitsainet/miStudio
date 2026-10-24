"""Template list filters, served by the list routes the Templates panels call.

The training and extraction template lists had a "Filters" button whose panel
read "Additional filters coming soon (architecture type, date range, etc.)" /
"Additional filters can be added here...". The filters are now real, and they
are applied SERVER-side: the lists are paginated (50 per page), so filtering in
the browser would filter one page and report it as the whole set.

  GET /api/v1/training-templates?encoder_type=&created_after=&created_before=
  GET /api/v1/extraction-templates?hook_type=&sort_by=&order=

Also fixed here (pre-existing): `sort_by` reached `getattr(Model, sort_by)`, so
an unknown name 500'd (or sorted by a relationship); it is now an allow-list,
422 at the route, with an id tiebreak so pages are stable.

NEGATIVE CONTROLS (2026-10-18, each restored by sha256):
  T1  drop the `created_after` filter append in the training service       -> 2 red
  T2  make `created_before` inclusive (<=)                                 -> 1 red
  T3  drop the 'standard' alias from the architecture filter               -> 1 red
  T4  drop `created_after=created_after,` from the route's service call    -> 2 red
  E1  drop the hook_type filter append in the extraction service           -> 1 red
  E2  drop `hook_type=hook_type,` from the route's service call            -> 1 red
  E3  ignore sort_by (always order by created_at)                          -> 2 red
"""

import datetime as dt

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.database import get_db
from src.main import app
from src.models.extraction_template import ExtractionTemplate
from src.models.training_template import TrainingTemplate

D = lambda day: dt.datetime(2026, 10, day, 12, 0, tzinfo=dt.timezone.utc)  # noqa: E731


@pytest_asyncio.fixture
async def api(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def per_request_session():
        async with maker() as session:
            yield session

    app.dependency_overrides[get_db] = per_request_session
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test/api/v1") as client:
        yield client
    app.dependency_overrides.clear()


async def _training_rows(session):
    rows = [
        ("legacy std", "standard", D(1)),
        ("saelens", "standard_saelens", D(5)),
        ("jump early", "jumprelu", D(3)),
        ("jump late", "jumprelu", D(10)),
        ("topk", "topk", D(7)),
    ]
    for name, enc, created in rows:
        session.add(TrainingTemplate(name=name, encoder_type=enc, hyperparameters={},
                                     dataset_ids=[], created_at=created, updated_at=created))
    await session.commit()


async def _extraction_rows(session):
    rows = [
        ("b residual", ["residual"], D(2)),
        ("a mlp+attn", ["mlp", "attention"], D(4)),
        ("c all", ["residual", "mlp", "attention"], D(6)),
    ]
    for name, hooks, created in rows:
        session.add(ExtractionTemplate(name=name, layer_indices=[1], hook_types=hooks,
                                       max_samples=10, batch_size=2, top_k_examples=5,
                                       created_at=created, updated_at=created))
    await session.commit()


def _names(body):
    return [t["name"] for t in body["data"]]


class TestTrainingTemplateFilters:
    async def test_architecture_filter(self, api, async_session):
        await _training_rows(async_session)
        r = await api.get("/training-templates", params={"encoder_type": "jumprelu"})
        assert r.status_code == 200
        assert sorted(_names(r.json())) == ["jump early", "jump late"]
        assert r.json()["pagination"]["total"] == 2

    async def test_standard_saelens_includes_the_legacy_alias(self, api, async_session):
        await _training_rows(async_session)
        r = await api.get("/training-templates", params={"encoder_type": "standard_saelens"})
        assert sorted(_names(r.json())) == ["legacy std", "saelens"]

    async def test_date_range_is_inclusive_from_exclusive_to(self, api, async_session):
        await _training_rows(async_session)
        r = await api.get("/training-templates", params={
            "created_after": D(3).isoformat(), "created_before": D(7).isoformat()})
        assert r.status_code == 200
        # D(3) is on the lower bound (kept); D(7) is on the upper bound (excluded).
        assert sorted(_names(r.json())) == ["jump early", "saelens"]
        assert r.json()["pagination"]["total"] == 2

    async def test_filters_combine(self, api, async_session):
        await _training_rows(async_session)
        r = await api.get("/training-templates", params={
            "encoder_type": "jumprelu", "created_after": D(4).isoformat()})
        assert _names(r.json()) == ["jump late"]

    async def test_unknown_sort_field_is_refused_not_a_500(self, api, async_session):
        r = await api.get("/training-templates", params={"sort_by": "__class__"})
        assert r.status_code == 422


class TestExtractionTemplateFilters:
    async def test_hook_type_filter_matches_any_listed_hook(self, api, async_session):
        await _extraction_rows(async_session)
        r = await api.get("/extraction-templates", params={"hook_type": "mlp"})
        assert r.status_code == 200
        assert sorted(_names(r.json())) == ["a mlp+attn", "c all"]
        assert r.json()["pagination"]["total"] == 2

    async def test_unknown_hook_type_is_refused(self, api, async_session):
        r = await api.get("/extraction-templates", params={"hook_type": "embedding"})
        assert r.status_code == 422

    @pytest.mark.parametrize("sort_by,order,expected", [
        ("name", "asc", ["a mlp+attn", "b residual", "c all"]),
        ("name", "desc", ["c all", "b residual", "a mlp+attn"]),
        ("created_at", "asc", ["b residual", "a mlp+attn", "c all"]),
        ("created_at", "desc", ["c all", "a mlp+attn", "b residual"]),
    ])
    async def test_sort_order(self, api, async_session, sort_by, order, expected):
        await _extraction_rows(async_session)
        r = await api.get("/extraction-templates", params={"sort_by": sort_by, "order": order})
        assert _names(r.json()) == expected

    async def test_unknown_sort_field_is_refused_not_a_500(self, api, async_session):
        r = await api.get("/extraction-templates", params={"sort_by": "hook_types"})
        assert r.status_code == 422
