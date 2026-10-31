"""The image build's fetch keeps ONLY the published bytes (Feature 040 S4; FTDD Appendix C.4, C.9 "Bundle identity").

`mistudio_rule_runner.fetch.run` reads URL, sha256 and size from the vendored contract; a snapshot one byte different
is refused (nothing is left where the bundle goes), a contract whose `bundle_digest` does not recompute from its own
parts is refused, and the verified bytes are kept with a precompiled `.cwasm` whose sidecar names them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mistudio_rule_runner import fetch
from mistudio_rule_runner.contract import CONTRACT_PATH


def test_the_published_snapshot_is_kept_and_precompiled(bundle_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    digest = fetch.run(out, snapshot_file=bundle_dir / "snapshot.wasm")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert digest == contract["guest"]["bundle"]["snapshot"]["sha256"]
    sidecar = json.loads((out / "snapshot.cwasm.json").read_text(encoding="utf-8"))
    assert sidecar["snapshot_sha256"] == digest and (out / "snapshot.cwasm").stat().st_size > 0


def test_a_one_byte_different_snapshot_is_refused(bundle_dir: Path, tmp_path: Path) -> None:
    data = bytearray((bundle_dir / "snapshot.wasm").read_bytes())
    data[-1] ^= 0x01
    forged = tmp_path / "forged.wasm"
    forged.write_bytes(bytes(data))
    out = tmp_path / "bundle"
    with pytest.raises(fetch.BundleError, match="sha256"):
        fetch.run(out, snapshot_file=forged, compile_=False)
    assert not (out / "snapshot.wasm").exists()


def test_a_contract_whose_bundle_digest_does_not_recompute_is_refused(bundle_dir: Path, tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["guest"]["bundle"]["snapshot"]["size"] += 1  # a hand edit with the old digest left in place
    edited = tmp_path / "contract.json"
    edited.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(fetch.BundleError, match="does not recompute"):
        fetch.run(tmp_path / "bundle", snapshot_file=bundle_dir / "snapshot.wasm", contract_path=edited,
                  compile_=False)
