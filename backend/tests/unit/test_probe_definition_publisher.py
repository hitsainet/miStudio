"""Publishing a definition to HuggingFace (033 4.1, 4.4).

⚠ THREE THINGS HERE WOULD EACH BE INVISIBLE UNTIL SOMEONE ELSE LOST DATA OR TRUST:

  * **the manifest merge.** A publisher that writes its own list silently deletes every probe
    someone else published to the same repo. Nobody notices until a consumer looks for a probe that
    used to be there.
  * **the README's "what has NOT been checked".** A reader who takes "detects on unseen tasks" for
    "is reliable on my traffic" will over-trust the probe, and the document is the only place that
    can say otherwise.
  * **the token.** A credential written to a row, a task result or a log line is readable by anyone
    who can list tasks.

MUTATION CONTROLS (each verified to fail this file):
  P1  the manifest is replaced instead of merged            → the merge tests
  P2  the merge drops entries it does not recognise         → the preservation test
  P3  the README's "not checked" section removed            → the honesty tests
  P4  the discovery tag removed from the front matter       → the tag test
  P5  `base_model` dropped from the front matter            → the base-model test
  P6  the token written into `published`                    → the secrecy tests
  P7  `upload_folder` sends fewer than three files          → the payload test
  P8  `published` is replaced rather than appended          → the history test
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.services.probe_definition_publisher import (
    DISCOVERY_TAG,
    MANIFEST_NAME,
    README_NAME,
    ProbePublishFailed,
    manifest_entry,
    merge_manifest,
    publish,
    render_readme,
)


def _definition(name="probe-a", rung=2, acknowledged=False):
    from tests.unit.test_probe_definition import definition

    body = definition()
    body["name"] = name
    body["evidence"]["rung"] = rung
    if rung < 2 or acknowledged:
        body["evidence"]["acknowledgement"] = {
            "by": "sean",
            "at": "2026-09-26T12:00:00+00:00",
            "reason": "exploratory monitor on a narrow concept; not used for gating",
        }
    return body


class TestTheManifestMergeNeverDeletes:
    def test_a_first_publication_starts_the_list(self):
        merged = merge_manifest(None, {"name": "a"})
        assert [item["name"] for item in merged["probes"]] == ["a"]
        assert merged["last_action"] == "appended"
        assert merged["kind"] == "mistudio.probe-manifest/v1"

    def test_a_new_name_is_appended_and_the_others_survive(self):
        existing = {"probes": [{"name": "a"}, {"name": "b"}]}
        merged = merge_manifest(existing, {"name": "c"})
        assert [item["name"] for item in merged["probes"]] == ["a", "b", "c"]
        assert merged["last_action"] == "appended"

    def test_the_same_name_is_REPLACED_not_duplicated(self):
        existing = {"probes": [{"name": "a", "rung": 1}, {"name": "b"}]}
        merged = merge_manifest(existing, {"name": "a", "rung": 3})
        names = [item["name"] for item in merged["probes"]]
        assert names == ["a", "b"]
        assert merged["last_action"] == "replaced"
        assert next(i for i in merged["probes"] if i["name"] == "a")["rung"] == 3

    def test_entries_this_publisher_did_not_write_are_PRESERVED_verbatim(self):
        """A repo accumulates probes from several sessions and possibly several people."""
        stranger = {"name": "somebody-elses", "invented_field": 42}
        merged = merge_manifest({"probes": [stranger]}, {"name": "mine"})
        assert stranger in merged["probes"]

    def test_a_malformed_manifest_is_treated_as_absent_rather_than_fatal(self):
        """Refusing to publish because someone hand-edited the manifest would strand the probe."""
        merged = merge_manifest({"probes": "not a list"}, {"name": "a"})
        assert [item["name"] for item in merged["probes"]] == ["a"]
        merged = merge_manifest(["a", "list"], {"name": "a"})
        assert [item["name"] for item in merged["probes"]] == ["a"]

    def test_the_order_is_stable(self):
        """Two publications in either order produce the same manifest, so a repo's history shows
        content changes rather than reordering."""
        one = merge_manifest({"probes": [{"name": "b"}]}, {"name": "a"})
        two = merge_manifest({"probes": [{"name": "a"}]}, {"name": "b"})
        assert [i["name"] for i in one["probes"]] == [i["name"] for i in two["probes"]]


class TestTheManifestEntrySummarises:
    def test_it_carries_what_a_consumer_scans_before_downloading(self):
        entry = manifest_entry(_definition(), "probe-a.probe.json")
        for field in ("name", "file", "base_model", "model_revision", "layer", "basis", "rule",
                      "rung", "rung_language", "mean_auroc", "evaluation_sets"):
            assert field in entry, field

    def test_the_mean_auroc_is_over_the_evaluations(self):
        body = _definition()
        body["evidence"]["evaluations"] = [
            dict(body["evidence"]["evaluations"][0], auroc=0.9),
            dict(body["evidence"]["evaluations"][0], auroc=0.7),
        ]
        assert manifest_entry(body, "x.json")["mean_auroc"] == 0.8

    def test_no_evaluations_gives_None_not_zero(self):
        """0.0 would read as "measured at chance"; None reads as "not measured"."""
        body = _definition(rung=1)
        body["evidence"]["evaluations"] = []
        assert manifest_entry(body, "x.json")["mean_auroc"] is None

    def test_an_acknowledged_export_says_so_in_the_summary(self):
        entry = manifest_entry(_definition(rung=1), "x.json")
        assert entry["acknowledged_below_rung2"] is True

    def test_a_rung_2_export_does_not_claim_an_acknowledgement(self):
        assert manifest_entry(_definition(rung=2), "x.json")["acknowledged_below_rung2"] is False


class TestTheReadmeIsHonest:
    def test_it_has_the_discovery_tag_in_its_front_matter(self):
        text = render_readme(_definition(), {"probes": []})
        assert text.startswith("---")
        assert f"  - {DISCOVERY_TAG}" in text

    def test_it_names_the_base_model(self):
        """`base_model` is what HuggingFace uses to link the probe to the model it reads."""
        text = render_readme(_definition(), {"probes": []})
        assert "base_model: meta-llama/Llama-3.1-8B-Instruct" in text

    def test_it_states_what_has_been_checked(self):
        text = render_readme(_definition(), {"probes": []})
        assert "## What has been checked" in text
        assert "resid_post" in text

    def test_it_states_what_has_NOT_been_checked(self):
        text = render_readme(_definition(), {"probes": []})
        assert "## What has NOT been checked" in text
        assert "Causality" in text, "the correlational caveat is the load-bearing one"
        assert "Your traffic" in text

    def test_a_rung_2_probe_is_told_it_has_no_judge_comparison(self):
        text = render_readme(_definition(rung=2), {"probes": []})
        assert "judge" in text.lower()

    def test_a_rung_3_probe_is_NOT_told_that(self):
        """The control: if the caveat were unconditional it would say nothing about this probe."""
        body = _definition(rung=3)
        body["evidence"]["judge"] = {
            "model": "Qwen2.5-7B-Instruct", "prompt_version": "stakes-rating/v1",
            "per_set_auroc": {"mt_balanced": 0.8},
        }
        text = render_readme(body, {"probes": []})
        assert "has not been measured against an LLM judge" not in text

    def test_a_below_rung_2_export_carries_its_acknowledgement_and_who_made_it(self):
        text = render_readme(_definition(rung=1), {"probes": []})
        assert "Held-out generalisation" in text
        assert "sean" in text
        assert "exploratory monitor" in text

    def test_the_evaluation_table_reports_the_confidence_interval(self):
        text = render_readme(_definition(), {"probes": []})
        assert "| set | distribution | AUROC | 95% CI |" in text
        assert "mt_balanced" in text

    def test_it_points_at_the_published_schema_and_the_tolerance(self):
        text = render_readme(_definition(), {"probes": []})
        assert "probe-definition-v1.json" in text
        assert "test_vectors" in text


class _FakeApi:
    """Records what would have been sent."""

    def __init__(self, existing_manifest=None, sha="c" * 40):
        self.existing_manifest = existing_manifest
        self.created = None
        self.uploaded = None
        self._sha = sha
        self.calls = []

    def create_repo(self, **kwargs):
        self.created = kwargs
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs
        self.calls.append(("upload_folder", kwargs))

    def model_info(self, repo_id):
        return SimpleNamespace(sha=self._sha)

    def hf_hub_download(self, repo_id, filename, repo_type="model"):
        if self.existing_manifest is None:
            raise FileNotFoundError(filename)
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(self.existing_manifest, handle)
        handle.close()
        return handle.name


class _Db:
    def __init__(self, probe):
        self.probe = probe
        self.commits = 0

    def query(self, _model):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.probe

    def commit(self):
        self.commits += 1


@pytest.fixture()
def built(tmp_path, monkeypatch):
    """A probe row whose definition is on disk."""
    from src.services import probe_definition_publisher as publisher

    path = tmp_path / "pm_x.probe.json"
    path.write_text(json.dumps(_definition()))
    monkeypatch.setattr(
        publisher, "settings",
        SimpleNamespace(resolve_data_path=lambda _p: path, data_dir=str(tmp_path)),
    )
    probe = SimpleNamespace(
        id="pm_x", definition_path=str(path), definition_sha256="d" * 64, published=[]
    )
    return _Db(probe), probe, tmp_path


class TestThePublishPayload:
    def test_it_uploads_exactly_three_files(self, built):
        db, probe, tmp_path = built
        api = _FakeApi()
        publish(db, "pm_x", repo_id="owner/probes", api=api, token="hf_secret")
        folder = Path(api.uploaded["folder_path"])
        names = sorted(item.name for item in folder.iterdir())
        assert names == sorted(["probe-a.probe.json", MANIFEST_NAME, README_NAME]), names

    def test_the_repo_is_created_private_by_default(self, built):
        db, _probe, _tmp = built
        api = _FakeApi()
        publish(db, "pm_x", repo_id="owner/probes", api=api, token="t")
        assert api.created["private"] is True
        assert api.created["exist_ok"] is True

    def test_the_commit_message_names_the_probe_and_its_rung(self, built):
        db, _probe, _tmp = built
        api = _FakeApi()
        publish(db, "pm_x", repo_id="owner/probes", api=api, token="t")
        assert "probe-a" in api.uploaded["commit_message"]
        assert "rung 2" in api.uploaded["commit_message"]

    def test_an_existing_manifest_is_merged_not_overwritten(self, built):
        db, _probe, _tmp = built
        api = _FakeApi(existing_manifest={"probes": [{"name": "someone-elses"}]})
        result = publish(db, "pm_x", repo_id="owner/probes", api=api, token="t")
        manifest = json.loads((Path(api.uploaded["folder_path"]) / MANIFEST_NAME).read_text())
        names = [item["name"] for item in manifest["probes"]]
        assert "someone-elses" in names and "probe-a" in names
        assert result["probes_in_repo"] == 2
        assert result["manifest_action"] == "appended"

    def test_the_result_carries_the_revision_and_a_url(self, built):
        db, _probe, _tmp = built
        api = _FakeApi(sha="e" * 40)
        result = publish(db, "pm_x", repo_id="owner/probes", api=api, token="t")
        assert result["revision"] == "e" * 40
        assert result["url"].startswith("https://huggingface.co/owner/probes/")

    def test_publication_history_is_APPENDED(self, built):
        """A probe published privately and later publicly has two publications; the first does not
        stop existing."""
        db, probe, _tmp = built
        publish(db, "pm_x", repo_id="owner/private", api=_FakeApi(), token="t", private=True)
        publish(db, "pm_x", repo_id="owner/public", api=_FakeApi(), token="t", private=False)
        assert [entry["repo_id"] for entry in probe.published] == ["owner/private", "owner/public"]
        assert [entry["private"] for entry in probe.published] == [True, False]

    def test_no_built_definition_is_refused(self, built):
        db, probe, _tmp = built
        probe.definition_path = None
        with pytest.raises(ProbePublishFailed) as caught:
            publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token="t")
        assert "build it before publishing" in str(caught.value)

    def test_a_missing_probe_is_refused(self, built):
        db, _probe, _tmp = built
        db.probe = None
        with pytest.raises(ProbePublishFailed):
            publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token="t")


class TestTheTokenIsNeverPersistedOrLogged:
    SECRET = "hf_thisisasecrettoken1234567890"

    def test_it_does_not_reach_the_row(self, built):
        db, probe, _tmp = built
        publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token=self.SECRET)
        assert self.SECRET not in json.dumps(probe.published)

    def test_it_does_not_reach_the_result(self, built):
        db, _probe, _tmp = built
        result = publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token=self.SECRET)
        assert self.SECRET not in json.dumps(result)

    def test_it_does_not_reach_the_uploaded_files(self, built):
        db, _probe, _tmp = built
        api = _FakeApi()
        publish(db, "pm_x", repo_id="owner/probes", api=api, token=self.SECRET)
        folder = Path(api.uploaded["folder_path"])
        for item in folder.iterdir():
            assert self.SECRET not in item.read_text(), item.name

    def test_it_does_not_reach_the_logs(self, built, caplog):
        import logging

        db, _probe, _tmp = built
        with caplog.at_level(logging.DEBUG):
            publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token=self.SECRET)
        assert self.SECRET not in caplog.text
        assert "***" in caplog.text, "the redaction should be visible, so a reader knows one was used"

    def test_the_caplog_assertion_could_fail(self, built, caplog):
        """The control: caplog must actually capture this module's records, or the secrecy test
        above passes because nothing was captured at all."""
        import logging

        db, _probe, _tmp = built
        with caplog.at_level(logging.INFO):
            publish(db, "pm_x", repo_id="owner/probes", api=_FakeApi(), token="t")
        assert "published to owner/probes" in caplog.text
