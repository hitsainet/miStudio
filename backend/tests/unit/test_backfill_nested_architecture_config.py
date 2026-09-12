"""The backfill must repair incomplete rows, reuse the real extractor, and
touch nothing else.

A data migration gets one attempt against real rows at deploy time, so its
behaviour is pinned here rather than discovered in production.
"""

import pytest

from src.db import architecture_backfill as mod


class TestItUsesTheProductionExtractor:
    def test_rebuild_delegates_rather_than_reimplementing(self, tmp_path):
        """Two copies of the field list drift; there must be one."""
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
        from src.ml.model_loader import extract_architecture_config

        Gemma3Config().save_pretrained(tmp_path)

        rebuilt = mod.describe_model(tmp_path)
        from transformers import AutoConfig

        expected = extract_architecture_config(
            AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)
        )
        assert rebuilt == expected

    def test_it_recovers_a_composite_layer_count(self, tmp_path):
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config

        cfg = Gemma3Config()
        cfg.save_pretrained(tmp_path)

        out = mod.describe_model(tmp_path)
        assert out["num_hidden_layers"] == cfg.get_text_config().num_hidden_layers
        assert "vision_config" in out["towers"]


class TestFindingTheConfig:
    def test_it_finds_a_huggingface_snapshot(self, tmp_path):
        snap = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        assert mod.config_dir(str(tmp_path)) == snap

    def test_it_finds_a_flat_layout(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert mod.config_dir(str(tmp_path)) == tmp_path

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert mod.config_dir(str(tmp_path / "nope")) is None

    def test_a_directory_without_a_config_is_not_an_error(self, tmp_path):
        assert mod.config_dir(str(tmp_path)) is None


class TestItOnlyTargetsBrokenRows:
    def test_the_query_filters_on_the_missing_layer_count(self):
        import inspect

        src = inspect.getsource(mod.affected_rows)
        assert "jsonb_exists" in src and "num_hidden_layers" in src, (
            "the backfill no longer restricts itself to rows missing a layer "
            "count, so it can overwrite a correct architecture_config"
        )

    def test_it_merges_rather_than_replaces(self):
        import inspect

        src = inspect.getsource(mod.backfill)
        assert "merged.update(rebuilt)" in src


class TestTheSqlMatchesTheRealSchema:
    """The gap that took the API down.

    The earlier tests here exercised the migration's HELPERS -- _rebuild and
    _config_dir -- and never its SQL, so a SELECT naming a column that does not
    exist passed every one of them. It reached production, the entrypoint
    refuses to serve without successful migrations, and the backend
    crashlooped for 34 minutes (11 restarts, 2026-08-25).

    A column name is only proven by a database. These run the migration's OWN
    query -- not a copy typed into the test, which would drift -- against the
    real schema.

    Deliberately NOT a source scrape: the first attempt at this test matched
    the column name inside the docstring explaining that the column does not
    exist, which is the recurring trap in this repo.
    """

    @pytest.mark.asyncio
    async def test_the_migrations_own_query_executes(self, async_session):
        rows = await async_session.run_sync(lambda sess: mod.affected_rows(sess))
        assert isinstance(rows, list)

    @pytest.mark.asyncio
    async def test_its_update_targets_a_real_column(self, async_session):
        """The UPDATE is only reached when a row needs repair, so drive the
        migration's OWN statement -- a copy typed here would pass while the
        real one named a column that does not exist."""
        await async_session.run_sync(
            lambda sess: mod.write_config(sess, "__no_such_model__", {})
        )

    def test_a_broken_select_cannot_stop_the_deploy(self):
        """Even if the SQL is wrong again, the API must still come up."""
        import inspect

        src = inspect.getsource(mod.backfill)
        after_call = src.split("affected_rows(conn)", 1)[1][:400]
        assert "except Exception" in after_call, (
            "the SELECT is unguarded again, so a bad query fails the migration "
            "and the entrypoint will not serve"
        )
        assert "return" in src.split("affected_rows(conn)", 1)[1][:700]


class TestAHeterogeneousModelSurvivesTheBackfill:
    """The end-to-end gap: every unit passed and the backfill repaired nothing.

    c4d8e1f60a92 ran in production, logged "repaired 0 model(s)", and the
    Training page stayed blank -- because one field of one tower refused to
    answer globally and took the whole description with it. Nothing here
    exercised a config that REFUSES, so nothing went red.
    """

    def test_a_refusing_field_does_not_lose_the_model(self, tmp_path):
        """Drive describe_model over a config whose text tower refuses."""
        from unittest.mock import patch

        class _Refusing:
            model_type = "gemma4_unified"
            num_hidden_layers = 48
            hidden_size = 3840
            vocab_size = 262144

            @property
            def num_key_value_heads(self):
                raise RuntimeError("per-layer attribute; no global value")

            def get_text_config(self, *a, **k):
                return self

        with patch("transformers.AutoConfig.from_pretrained", return_value=_Refusing()):
            described = mod.describe_model(tmp_path)

        assert described["num_hidden_layers"] == 48, (
            "the model was dropped over a single per-layer attribute, which is "
            "how the backfill repaired 0 rows in production"
        )
        assert described["heterogeneous_layers"] is True
        assert "num_key_value_heads" not in described

    def test_the_row_is_actually_written(self, tmp_path):
        """A description that is never stored fixes nothing."""
        from unittest.mock import MagicMock, patch

        class _Refusing:
            model_type = "gemma4_unified"
            num_hidden_layers = 48

            @property
            def num_key_value_heads(self):
                raise RuntimeError("per-layer attribute; no global value")

            def get_text_config(self, *a, **k):
                return self

        (tmp_path / "config.json").write_text("{}")

        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            ("m_b55c6926", str(tmp_path), {"model_type": "gemma4_unified"})
        ]

        with patch("transformers.AutoConfig.from_pretrained", return_value=_Refusing()):
            repaired = mod.backfill(conn)

        assert repaired == 1, "the backfill reported no repair for a fixable row"

        written = [
            c for c in conn.execute.call_args_list
            if "UPDATE models" in str(c.args[0])
        ]
        assert written, "no UPDATE was issued"
        payload = written[-1].args[1]
        assert payload["mid"] == "m_b55c6926"
        assert '"num_hidden_layers": 48' in payload["cfg"]
