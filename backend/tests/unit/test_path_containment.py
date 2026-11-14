"""The shared containment check, and every CodeQL py/path-injection sink routed through it.

Security alerts 2026-10-17 (`0xcc/reviews/security_alerts_2026-10-17.md`). Each class below
drives a REAL sink with a path that leaves its root — by `..`, by an absolute segment, or by
a symlink inside the root pointing out — and requires a refusal. The symlink cases matter
most: a string-only normaliser cannot see them, and they are what the helper adds over the
checks that were already there.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.path_containment import (
    PathEscape,
    child_of,
    contained,
    contained_in_any,
    safe_segment,
)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def outside(tmp_path):
    o = tmp_path / "outside"
    o.mkdir()
    (o / "secret.txt").write_text("not yours")
    return o


class TestContained:
    def test_a_child_is_returned_as_its_real_path(self, root):
        (root / "a").mkdir()
        assert contained(root, root / "a") == Path(os.path.realpath(root / "a"))

    def test_the_root_itself_is_contained(self, root):
        assert contained(root, root) == Path(os.path.realpath(root))

    def test_dotdot_is_refused(self, root, outside):
        with pytest.raises(PathEscape):
            contained(root, root / ".." / "outside")

    def test_an_absolute_path_elsewhere_is_refused(self, root, outside):
        with pytest.raises(PathEscape):
            contained(root, outside / "secret.txt")

    def test_a_sibling_sharing_the_prefix_is_refused(self, tmp_path, root):
        """`/x/root-evil` starts with `/x/root` as a string; it is not inside it."""
        evil = tmp_path / "root-evil"
        evil.mkdir()
        with pytest.raises(PathEscape):
            contained(root, evil)

    def test_a_symlink_inside_that_points_out_is_refused(self, root, outside):
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscape):
            contained(root, root / "link" / "secret.txt")

    def test_a_symlink_inside_that_stays_inside_is_allowed(self, root):
        (root / "real").mkdir()
        (root / "alias").symlink_to(root / "real", target_is_directory=True)
        assert contained(root, root / "alias") == Path(os.path.realpath(root / "real"))

    def test_it_is_a_value_error_so_existing_handlers_refuse_it(self):
        assert issubclass(PathEscape, ValueError)


class TestContainedInAny:
    def test_the_first_root_that_holds_it_wins(self, tmp_path, root):
        other = tmp_path / "other"
        other.mkdir()
        (other / "x").mkdir()
        assert contained_in_any([root, other], other / "x") == Path(os.path.realpath(other / "x"))

    def test_outside_every_root_is_refused(self, root, outside):
        with pytest.raises(PathEscape):
            contained_in_any([root], outside)


class TestSegments:
    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "/abs", "a\\b", "a\x00b"])
    def test_not_one_segment_is_refused(self, bad):
        with pytest.raises(PathEscape):
            safe_segment(bad)

    @pytest.mark.parametrize("good", ["train_1a2b", "gemma-2-2b", "x.superseded", "..."])
    def test_one_segment_passes(self, good):
        assert safe_segment(good) == good

    def test_child_of_checks_shape_and_real_path(self, root, outside):
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscape):
            child_of(root, "..")
        with pytest.raises(PathEscape):
            child_of(root, "link")
        assert child_of(root, "fine") == Path(os.path.realpath(root)) / "fine"


# --------------------------------------------------------------------- the sinks


class TestExtractionDirectory:
    """#49/#50/#51: `delete_extraction` hands this directory to rmtree."""

    def _service(self, activations):
        from src.services.activation_service import ActivationService

        svc = ActivationService.__new__(ActivationService)
        svc.activations_dir = activations
        return svc

    def test_an_id_that_is_a_symlink_out_of_the_root_is_refused(
        self, tmp_path, outside, monkeypatch
    ):
        from src.core.config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        activations = tmp_path / "activations"
        activations.mkdir()
        (activations / "extr_evil").symlink_to(outside, target_is_directory=True)

        svc = self._service(activations)
        with pytest.raises(ValueError):
            svc._extraction_dir("extr_evil")
        with pytest.raises(ValueError):
            svc.delete_extraction("extr_evil")
        assert (outside / "secret.txt").exists(), "rmtree followed the link out of the root"

    def test_a_real_extraction_is_still_found(self, tmp_path, monkeypatch):
        from src.core.config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        activations = tmp_path / "activations"
        (activations / "extr_ok").mkdir(parents=True)
        svc = self._service(activations)
        assert svc._extraction_dir("extr_ok") == Path(os.path.realpath(activations / "extr_ok"))
        assert svc.delete_extraction("extr_ok") is True
        assert not (activations / "extr_ok").exists()


class TestJLensArtifactSlugs:
    """#38/#39/#40/#41/#44-#48/#64: a slug of `..` named the root's PARENT, which
    `restore_superseded` renames."""

    def test_slug_for_refuses_dot_and_dotdot(self):
        from src.services.jlens_artifact_service import slug_for

        for repo in ("..", "org/..", "."):
            with pytest.raises(ValueError):
                slug_for(repo)
        assert slug_for("google/Gemma-2-2B") == "gemma-2-2b"

    def test_find_never_looks_outside_the_root(self, tmp_path, monkeypatch):
        from src.services import jlens_artifact_service as mod

        root = tmp_path / "jlens"
        root.mkdir()
        (tmp_path / "x_jacobian_lens.pt").write_bytes(b"")  # an "artifact" in the PARENT
        monkeypatch.setattr(mod, "slug_for", lambda repo_id: "..")
        with pytest.raises(PathEscape):
            mod.JLensArtifactService(root).find("anything")

    def test_find_refuses_a_slug_directory_symlinked_out(self, tmp_path, outside):
        from src.services.jlens_artifact_service import JLensArtifactService

        root = tmp_path / "jlens"
        root.mkdir()
        (outside / "m_jacobian_lens.pt").write_bytes(b"")
        (root / "model").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscape):
            JLensArtifactService(root).find("org/model")

    @pytest.mark.parametrize("slug", ["..", ".", "a/b", "../root"])
    def test_restore_refuses_a_slug_that_is_not_one_segment(self, tmp_path, slug):
        from src.services.jlens_artifact_service import JLensArtifactService

        root = tmp_path / "jlens"
        root.mkdir()
        with pytest.raises(PathEscape):
            JLensArtifactService(root).restore_superseded(slug)
        assert root.is_dir() and not any(root.iterdir())

    def test_restore_refuses_an_archive_symlinked_out(self, tmp_path, outside):
        from src.services.jlens_artifact_service import JLensArtifactService

        root = tmp_path / "jlens"
        root.mkdir()
        (root / "m.superseded").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscape):
            JLensArtifactService(root).restore_superseded("m")
        assert (root / "m.superseded").is_symlink(), "the link was renamed into service"

    def test_the_route_answers_400_not_500(self, monkeypatch, tmp_path):
        from fastapi import HTTPException

        from src.api.v1.endpoints import jlens as endpoint
        from src.services.jlens_artifact_service import JLensArtifactService

        root = tmp_path / "jlens"
        root.mkdir()
        monkeypatch.setattr(endpoint, "_service", lambda: JLensArtifactService(root))
        with pytest.raises(HTTPException) as caught:
            asyncio.run(endpoint.restore_superseded(".."))
        assert caught.value.status_code == 400


class TestSAEImportFromFile:
    """#12-#15/#55-#57/#61: the import copies whatever `file_path` names into SAE storage.
    `resolve_user_path` is string-only, so a symlink in the data root reached past it."""

    def test_a_symlink_in_the_data_root_that_leaves_it_is_refused(
        self, tmp_path, outside, monkeypatch
    ):
        from src.core.config import settings
        from src.services.sae_manager_service import SAEManagerService

        data = tmp_path / "data"
        data.mkdir()
        monkeypatch.setattr(settings, "data_dir", data)
        monkeypatch.setattr(settings, "hf_cache_dir", data / "hf")
        (data / "saes_in").symlink_to(outside, target_is_directory=True)

        request = SimpleNamespace(file_path="saes_in", layer=None)
        with pytest.raises(ValueError, match="outside"):
            asyncio.run(SAEManagerService.import_from_file(db=None, request=request))
        assert not (data / "saes").exists() or not any((data / "saes").rglob("secret.txt"))


class TestTrainingDirectories:
    """#37/#58: the training id arrives in a URL and these paths are listed, read and written."""

    @pytest.mark.parametrize("training_id", ["..", "../../etc", "a/b", ""])
    def test_an_id_that_is_not_one_segment_is_refused(self, tmp_path, monkeypatch, training_id):
        from src.core.config import settings
        from src.services import training_finalize_service as tfs

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        with pytest.raises(PathEscape):
            tfs.community_format_dir(training_id)
        with pytest.raises(PathEscape):
            tfs.list_checkpoint_steps(training_id)
        with pytest.raises(PathEscape):
            tfs.full_length_export_step(training_id, 100)

    def test_a_training_dir_symlinked_out_is_refused(self, tmp_path, outside, monkeypatch):
        from src.core.config import settings
        from src.services import training_finalize_service as tfs

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "trainings").mkdir()
        (tmp_path / "trainings" / "train_x").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathEscape):
            tfs.list_checkpoint_steps("train_x")

    def test_a_real_training_is_still_read(self, tmp_path, monkeypatch):
        from src.core.config import settings
        from src.services import training_finalize_service as tfs

        monkeypatch.setattr(settings, "data_dir", tmp_path)
        (tmp_path / "trainings" / "train_ok" / "checkpoints" / "checkpoint_40").mkdir(parents=True)
        assert tfs.list_checkpoint_steps("train_ok") == [40]
