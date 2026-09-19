"""
Two on-disk checkpoint shapes exist, and both are real.

The conformance spec (`0xcc/brds/neuronpedia-jlens-conformance.md` §2.2) — which the
reference implementation and every lens published to HuggingFace follow — is a
WRAPPER, `{"J": {...}, "source_layers", "n_prompts", "d_model"}`. Every artifact
this project has written so far is the BARE map `{layer: Tensor}`.

Reading only the bare map is what made a real published lens fail with
`ValueError: invalid literal for int() with base 10: 'J'`. Reading only the wrapper
would strand every artifact currently in the registry. So both are first-class, and
anything else is refused BY NAME rather than guessed at — a wrong unwrap produces a
dict of square matrices that passes STRUCTURAL, ENVELOPE and NAMING and reads out
plausible nonsense.

MUTATION CONTROLS (each must turn this file red):
  * drop the wrapper branch              -> "the upstream WRAPPER is unwrapped"
  * make the wrapper branch unconditional -> "a BARE layer map still loads"
  * return the payload unchanged on the fallback -> "REFUSED and NAMES its keys"
  * pick either side when both present   -> "BOTH J and layer keys is refused"
  * skip the source_layers cross-check   -> "source_layers disagreeing is refused"
  * normalise in the acquire worker instead of `_load_payload`
                                          -> "EVERY READER sees the normalised shape"
"""

from __future__ import annotations

import pytest
import torch

from src.services.jlens_artifact_service import (
    ArtifactRef,
    JLensArtifactService,
    PayloadShapeError,
    normalise_payload,
)

#: Deliberately NOT equal to the layer count, and not square-adjacent to it.
#:
#: With d_model == n_layers a transposed or wrongly-axised unwrap produces
#: matrices of exactly the expected shape and every assertion passes by
#: coincidence. Three layers of 5x5 cannot agree with each other by accident.
D_MODEL = 5
LAYERS = (0, 1, 2)


def _matrices(layers=LAYERS, d_model=D_MODEL):
    # Each layer gets a DISTINGUISHABLE matrix, so a test can tell not merely
    # that something came back but that the right thing came back under the
    # right key.
    return {l: torch.full((d_model, d_model), float(l + 1)) for l in layers}


def _wrapped(layers=LAYERS, **extra):
    payload = {
        "J": _matrices(layers),
        "d_model": D_MODEL,
        "n_prompts": 337,
        "source_layers": list(layers),
    }
    payload.update(extra)
    return payload


class TestBothShapesLoad:
    def test_the_upstream_WRAPPER_is_unwrapped(self):
        out = normalise_payload(_wrapped())
        assert sorted(out) == list(LAYERS)
        assert out[1][0][0] == 2.0, "layers came back under the wrong keys"

    def test_a_BARE_layer_map_still_loads(self):
        """Every artifact currently on disk is this shape.

        A wrapper-only implementation destroys the existing registry, which is a
        far worse outcome than failing to read a new file.
        """
        out = normalise_payload(_matrices())
        assert sorted(out) == list(LAYERS)
        assert out[2][0][0] == 3.0

    def test_STRING_layer_keys_emit_integers_in_the_WRAPPER(self):
        """Spec §2.2: "Layer keys are coerced with `int()`, so string keys
        survive, but emit integers." Downstream indexes by int."""
        out = normalise_payload({"J": {"0": torch.zeros(D_MODEL, D_MODEL)}})
        assert list(out) == [0]
        assert isinstance(next(iter(out)), int)

    def test_STRING_layer_keys_emit_integers_in_the_BARE_MAP_TOO(self):
        """The bare path needs its OWN string fixture.

        With int keys the coercion is a no-op, so `return obj` — dropping it
        entirely — passed every test above: the wrapper test covered the wrapper
        branch and the bare tests all used int keys. Two fixtures agreeing by
        construction, hiding a branch that is the ONLY thing standing between a
        string-keyed artifact and `payload[0]` raising KeyError downstream.

        MUTATION CONTROL: `return obj` on the bare path and this fails.
        """
        out = normalise_payload({"1": torch.zeros(D_MODEL, D_MODEL)})
        assert list(out) == [1]
        assert isinstance(next(iter(out)), int), (
            "a string-keyed bare map came back with string keys; every reader "
            "indexes by int"
        )


class TestAnythingElseIsRefusedByName:
    def test_an_unrecognised_payload_is_REFUSED_and_NAMES_its_keys(self):
        """"Unrecognised checkpoint" sends a reader to the wrong file."""
        with pytest.raises(PayloadShapeError) as exc:
            normalise_payload({"state_dict": {}, "epoch": 3})
        message = str(exc.value)
        assert "state_dict" in message and "epoch" in message, message

    def test_BOTH_a_J_block_and_layer_keys_is_refused(self):
        """Two candidate lenses in one file. Picking either is a guess."""
        with pytest.raises(PayloadShapeError, match="BOTH"):
            normalise_payload({"J": _matrices(), 0: torch.zeros(D_MODEL, D_MODEL)})

    def test_source_layers_DISAGREEING_with_J_is_refused(self):
        """A1 requires equality. A disagreement means the file was assembled
        from parts and one of them is stale — possibly the matrices."""
        with pytest.raises(PayloadShapeError, match="source_layers"):
            normalise_payload(_wrapped(source_layers=[0, 1, 2, 3]))

    def test_source_layers_ABSENT_is_fine(self):
        """Spec §2.2 makes it optional. Absent is not a disagreement."""
        payload = _wrapped()
        del payload["source_layers"]
        assert sorted(normalise_payload(payload)) == list(LAYERS)

    def test_a_non_dict_checkpoint_is_refused(self):
        with pytest.raises(PayloadShapeError, match="Tensor|list"):
            normalise_payload([torch.zeros(2, 2)])

    def test_an_EMPTY_checkpoint_is_refused(self):
        """`{}` would otherwise normalise to `{}` and read as a lens covering no
        layers, which `fitted_layers` documents as meaning UNKNOWN."""
        with pytest.raises(PayloadShapeError, match="empty"):
            normalise_payload({})

    def test_an_empty_J_block_is_refused(self):
        with pytest.raises(PayloadShapeError, match="non-empty"):
            normalise_payload({"J": {}, "d_model": D_MODEL})


class TestTheNormalisationIsWhereEveryReaderSeesIt:
    """The placement test, and the reason this lives in `_load_payload`.

    Normalising inside the acquire worker instead would leave `validate`,
    `check_structural`, `_coverage_delta`, `load_for_readout`, both semantic-check
    sites and the endpoint's `sorted(int(k) for k in payload)` still raising
    `ValueError('J')` on a file that is on disk and published.

    MUTATION CONTROL: move `normalise_payload` out of `_load_payload` and this
    fails.
    """

    @staticmethod
    def _artifact(tmp_path, payload):
        directory = tmp_path / "model"
        directory.mkdir()
        lens = directory / "model_jacobian_lens.pt"
        torch.save(payload, lens)
        config = directory / "config.yaml"
        config.write_text("model: org/model\n")
        return JLensArtifactService(tmp_path), ArtifactRef(
            slug="model",
            directory=directory,
            lens_path=lens,
            config_path=config,
        )

    def test_a_WRAPPED_file_on_disk_reads_as_layers(self, tmp_path):
        service, ref = self._artifact(tmp_path, _wrapped())
        payload = service._load_payload(ref)  # noqa: SLF001 - same package concern
        assert payload is not None, "a conformant published lens failed to load"
        assert sorted(payload) == list(LAYERS)

    def test_the_ENDPOINT_expression_works_on_a_wrapped_file(self, tmp_path):
        """`validate_artifact` does exactly this, and it is what raised
        `ValueError: invalid literal for int() with base 10: 'J'` against a real
        published lens."""
        service, ref = self._artifact(tmp_path, _wrapped())
        payload = service._load_payload(ref)  # noqa: SLF001
        assert sorted(int(k) for k in payload) == list(LAYERS)

    def test_VALIDATE_passes_on_a_wrapped_file(self, tmp_path):
        """The whole point: the existing validator already works, once the file
        can be read at all."""
        from src.services.jlens_validation import CheckClass, CheckStatus

        service, ref = self._artifact(tmp_path, _wrapped())
        report = service.validate(
            ref, d_model=D_MODEL, expected_layers=LAYERS, n_vocab=32
        )
        by_class = {r.check: r.status for r in report.results}
        assert by_class[CheckClass.STRUCTURAL] is CheckStatus.PASS, by_class
        assert by_class[CheckClass.ENVELOPE] is CheckStatus.PASS, by_class

    def test_an_UNREADABLE_shape_is_reported_as_a_failure_not_a_crash(self, tmp_path):
        """`_load_payload` returns None, which `validate` turns into a STRUCTURAL
        FAIL. A raised PayloadShapeError here would take down every caller
        including the artifact LISTING, which must survive one bad directory."""
        service, ref = self._artifact(tmp_path, {"state_dict": {}, "epoch": 1})
        assert service._load_payload(ref) is None  # noqa: SLF001
        assert service.list_artifacts(), "one unreadable artifact broke discovery"
