"""A probe dataset may be registered WITH a miDataworks dataset-version manifest (Feature 034,
FR-26; SC-10; XR-1).

`POST /api/v1/probe-monitors/datasets` with `dataset_version_manifest`:
  valid → recorded on the row and returned by create and list;
  invalid → 422 naming the failing JSON path;
  manifest without `split` → 422;
  split not described by the manifest → 409 listing the manifest's splits;
  manifest rows ≠ downloaded split rows → 409 naming both counts, compared BEFORE the keyword
  filter (a filter that drops rows still passes);
  no manifest → exactly as before.

Real code end to end: a real Arrow `DatasetDict` written with `save_to_disk`, the real
`load_columns`, the real endpoint, PostgreSQL. The manifest fixture is checked against the
committed schema by the first test, so a schema change breaks it visibly rather than silently.

Controls (recorded 2026-10-16, task 3.6; each red, each restore verified):
  * delete the count comparison (`total != expected_rows` → `False`) →
    `test_a_row_count_MISMATCH_is_refused_naming_both` red;
  * `!=` → `<` SURVIVED the first version of this file (only "manifest states more" was tested);
    `test_a_manifest_stating_FEWER_rows_is_refused_too` added, and the same mutation is now red;
  * `!=` → `>` → `test_a_row_count_MISMATCH_is_refused_naming_both` red;
  * delete the split-required check → `test_a_manifest_without_a_split_is_refused` red;
  * rename the field on `ProbeDatasetCreate` → `test_the_field_is_in_the_SERVED_openapi` red.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from src.models.dataset import Dataset, DatasetStatus
from src.services import dataset_version_manifest as manifests

H = "0" * 64
def manifest(splits=None, published=True):
    splits = splits or {"train": 120, "test": 80}
    return {
        "kind": "midataworks.dataset-version", "schema_version": "1",
        "producer": {"app": "miDataworks", "build": "b1", "generated_at": "2026-10-16T00:00:00Z"},
        "version": {"dataset": "stakes", "version_id": "dv_1", "number": 3,
                    "parent_version_id": None, "created_at": "2026-10-16T00:00:00Z",
                    "created_by_origin": "operator",
                    "recipe": {"format": "dw.recipe/v1", "sha256": H}, "seed": 1,
                    "row_key_scheme": "k"},
        "target": {"kind": "dataset", "dataset_target_type": "detector", "trl_type": None,
                   "trl_version": None, "detector_role": None, "miforge_set_kind": None,
                   "contract_reference": None},
        "sources": [{"source_id": "src_1",
                     "origin": {"kind": "hub", "repo_id": "org/raw", "config": None,
                                "split": "train", "revision": "b" * 40},
                     "licence": {"displayed": "MIT", "raw": "mit", "origin": "card_data",
                                 "licence_class": "permits_redistribution", "table_version": 1},
                     "terms_status": "not_recorded", "gated": False, "extensions": {}}],
        "content": {"content_kind": "rows", "columns": [],
                    "splits": [{"name": n, "path": f"data/{n}.parquet", "rows": r, "bytes": 1,
                                "sha256": H, "logical_digest": H, "label_counts": {},
                                "held_out": False, "evaluation_only": False, "extensions": {}}
                               for n, r in splits.items()],
                    "projection": {"label_resolver": "dw.effective-label/v1", "label_column": None,
                                   "omitted_excluded": 0, "omitted_flagged_unresolved": 0,
                                   "overrides_applied": 0},
                    "labelers": [], "calibration": []},
        "caveats": [],
        "publication": ({"state": "published", "repo_id": "org/stakes", "repo_type": "dataset",
                         "visibility": "private", "commit": "a" * 40,
                         "published_at": "2026-10-16T00:00:00Z",
                         "verification": {"result": "hashes_match", "files_checked": 2,
                                          "checked_at": "2026-10-16T00:00:00Z"}}
                        if published else None),
        "extensions": {},
    }


@pytest.fixture(scope="module")
def arrow_dict(tmp_path_factory):
    """`train` (120 rows) and `test` (80 rows): `text`, `label` ("yes"/"no").

    Half of each class carries the word `keep`, so a keyword filter can halve a split and still
    leave the 20-per-class floor `build_view` enforces."""
    from datasets import Dataset as HFDataset, DatasetDict

    def split(n):
        return HFDataset.from_dict({
            "text": [("risk" if i % 2 == 0 else "calm") + (" keep" if i % 4 < 2 else "") + f" {i}"
                     for i in range(n)],
            "label": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        })

    path = tmp_path_factory.mktemp("dw034") / "arrow"
    DatasetDict({"train": split(120), "test": split(80)}).save_to_disk(str(path))
    return path


@pytest_asyncio.fixture
async def dataset_id(async_session, arrow_dict):
    row = Dataset(id=uuid.uuid4(), name="stakes", source="local", status=DatasetStatus.READY,
                  raw_path=str(arrow_dict), extra_metadata={})
    async_session.add(row)
    await async_session.commit()
    return str(row.id)


def _body(dataset_id, **overrides):
    body = {
        "name": "stakes-test", "dataset_id": dataset_id, "split": "test",
        "input_column": "text", "label_column": "label",
        "label_mapping": {"yes": "positive", "no": "negative"},
        "dataset_version_manifest": manifest(),
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not ...}


URL = "/api/v1/probe-monitors/datasets"


def test_the_fixture_is_a_VALID_manifest_of_the_committed_schema():
    facts = manifests.validate(manifest())
    assert facts.split_names() == ["train", "test"]


def test_every_fact_path_exists_in_the_schema():
    """A renamed field must fail HERE, not as a silent `null` in production."""
    schema = manifests.load_schema()
    defs = schema["$defs"]

    def resolve(node):
        while "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[1]]
        return node

    def candidates(node):
        node = resolve(node)
        options = node.get("oneOf") or node.get("anyOf")
        if options:
            return [c for o in options for c in candidates(o)]
        return [node]

    for name, path in manifests.FACT_PATHS.items():
        nodes = [schema]
        for key in path:
            nxt = []
            for node in nodes:
                for c in candidates(node):
                    if key == "[]":
                        if "items" in c:
                            nxt.append(c["items"])
                    elif key in c.get("properties", {}):
                        nxt.append(c["properties"][key])
            assert nxt, f"{name}: {'.'.join(path)} does not exist in the schema at {key!r}"
            nodes = nxt


class TestRegistration:
    @pytest.mark.asyncio
    async def test_a_valid_manifest_is_RECORDED_and_returned(self, client, dataset_id):
        created = await client.post(URL, json=_body(dataset_id))
        assert created.status_code in (200, 201), created.text
        record = created.json()["dataset_version"]
        assert record["kind"] == "mistudio.verified-dataset-version/1"
        assert (record["version_id"], record["version_number"], record["dataset_name"]) == (
            "dv_1", 3, "stakes")
        assert (record["split"], record["manifest_rows"], record["downloaded_rows"]) == (
            "test", 80, 80)
        assert (record["repository"], record["commit"]) == ("org/stakes", "a" * 40)
        assert record["manifest_sha256"] == manifests.canonical_sha256(manifest())
        listed = await client.get(URL)
        rows = listed.json()
        rows = rows.get("datasets", rows) if isinstance(rows, dict) else rows
        mine = [r for r in rows if r["id"] == created.json()["id"]]
        assert mine and mine[0]["dataset_version"] == record

    @pytest.mark.asyncio
    async def test_an_UNPUBLISHED_version_records_null_commit_never_a_default(
        self, client, dataset_id
    ):
        doc = manifest()
        doc["publication"] = {"state": "in_repository", "repo_id": "org/stakes",
                              "repo_type": "dataset"}
        created = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        assert created.status_code in (200, 201), created.text
        record = created.json()["dataset_version"]
        assert record["commit"] is None and record["repository"] == "org/stakes"
        doc["publication"] = None
        created = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        record = created.json()["dataset_version"]
        assert record["commit"] is None and record["repository"] is None

    @pytest.mark.asyncio
    async def test_an_INVALID_manifest_is_refused_naming_the_path(self, client, dataset_id):
        doc = manifest()
        doc["version"]["number"] = 0
        refused = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        assert refused.status_code == 422
        assert "$.version.number" in refused.text

    @pytest.mark.asyncio
    async def test_a_manifest_without_a_split_is_refused(self, client, dataset_id):
        refused = await client.post(URL, json=_body(dataset_id, split=None))
        assert refused.status_code == 422
        assert "name the split" in refused.text

    @pytest.mark.asyncio
    async def test_a_split_the_manifest_does_not_describe_is_refused(self, client, dataset_id):
        doc = manifest(splits={"train": 120})
        refused = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        assert refused.status_code == 409
        assert "'test'" in refused.text and "train" in refused.text

    @pytest.mark.asyncio
    async def test_a_row_count_MISMATCH_is_refused_naming_both(self, client, dataset_id):
        doc = manifest(splits={"train": 120, "test": 81})
        refused = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        assert refused.status_code == 409
        assert refused.json()["detail"] == (
            "split 'test': manifest states 81 rows, downloaded split has 80"
        )

    @pytest.mark.asyncio
    async def test_a_manifest_stating_FEWER_rows_is_refused_too(self, client, dataset_id):
        """Review round, 2026-10-16: mutation `!=` → `<` SURVIVED with only the
        manifest-states-more case above. A download LARGER than the manifest says is just as
        unverified — rows nobody published — so both directions are pinned.
        Negative control: `!=` → `<` → this test red."""
        doc = manifest(splits={"train": 120, "test": 79})
        refused = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=doc))
        assert refused.status_code == 409
        assert "manifest states 79 rows, downloaded split has 80" in refused.json()["detail"]

    @pytest.mark.asyncio
    async def test_the_count_is_compared_BEFORE_the_keyword_filter(self, client, dataset_id):
        """The manifest describes the split FILE; miStudio's own filter narrows after. A filter
        that drops half the rows must still pass a manifest stating the full count."""
        body = _body(dataset_id, keyword_filter={"terms": ["keep"], "mode": "any"},
                     label_mapping={"yes": "positive", "no": "negative"})
        created = await client.post(URL, json=body)
        assert created.status_code in (200, 201), created.text
        assert created.json()["counts"]["filtered_out"] == 40
        assert created.json()["dataset_version"]["downloaded_rows"] == 80

    @pytest.mark.asyncio
    async def test_WITHOUT_a_manifest_nothing_changes(self, client, dataset_id):
        created = await client.post(URL, json=_body(dataset_id, dataset_version_manifest=...,
                                                     split=None))
        assert created.status_code in (200, 201), created.text
        assert created.json()["dataset_version"] is None


def test_the_field_is_in_the_SERVED_openapi():
    """XR-1: miDataworks detects support by reading this property from miStudio's OpenAPI
    document (served at /api/openapi.json). A rename silently turns its manifest send off."""
    from src.main import app

    schemas = app.openapi()["components"]["schemas"]
    assert "dataset_version_manifest" in schemas["ProbeDatasetCreate"]["properties"]
    assert "dataset_version" in schemas["ProbeDatasetResponse"]["properties"]
