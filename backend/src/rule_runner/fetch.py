"""Fetch the published guest snapshot BY DIGEST and precompile it (Feature 040 S4; FTDD Appendix C.4; D41 S0).

The image build (`rule_runner/Dockerfile`) and the `rule-runner-tests` CI job run ::

    python -m mistudio_rule_runner.fetch --out "$RUNNER_BUNDLE_DIR"      # in the image
    python -m src.rule_runner.fetch --out "$RUNNER_BUNDLE_DIR"           # from the backend tree

It reads the snapshot's URL, sha256 and size from the VENDORED contract (`guest.bundle.snapshot`; there is no
literal digest anywhere else), refuses unless the contract's own ``bundle_digest`` recomputes from its parts, and
keeps the download only if its bytes hash to the contract's value and its size is the contract's. It then
precompiles ``snapshot.cwasm`` for this Wasmtime and CPU, with a sidecar naming the snapshot, the stack size and
the Wasmtime version (`bundle._module` uses it only on an exact match and otherwise compiles the verified
snapshot). Nothing fetched is ever committed (`test_no_wasm_is_tracked`). Exit status 0 only when every digest
matched. Standard library + ``wasmtime`` (for ``--compile``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .contract import CONTRACT_PATH, bundle_digest


class BundleError(Exception):
    """A digest did not match: the bundle must not be used."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_contract_bundle(contract: Dict[str, Any]) -> Dict[str, Any]:
    bundle: Dict[str, Any] = contract["guest"]["bundle"]
    expected = bundle_digest(bundle)
    if bundle.get("bundle_digest") != expected:
        raise BundleError(f"contract bundle_digest {bundle.get('bundle_digest')!r} does not recompute ({expected})")
    return bundle


def _verify(path: Path, sha256: str, size: int) -> None:
    actual = sha256_file(path)
    if actual != sha256:
        raise BundleError(f"sha256 {actual} at {path}; the contract says {sha256}")
    if path.stat().st_size != size:
        raise BundleError(f"{path} is {path.stat().st_size} bytes; the contract says {size}")


def precompile(snapshot: Path, out: Path, max_wasm_stack: int) -> Path:
    """``snapshot.cwasm`` for THIS host's Wasmtime and CPU, configured exactly as `host.Host.engine_for`."""
    from importlib.metadata import version

    import wasmtime

    config = wasmtime.Config()
    config.consume_fuel = True
    config.epoch_interruption = True
    config.max_wasm_stack = max_wasm_stack
    module = wasmtime.Module.from_file(wasmtime.Engine(config), str(snapshot))
    target = out / "snapshot.cwasm"
    target.write_bytes(module.serialize())
    (out / "snapshot.cwasm.json").write_text(json.dumps(
        {"snapshot_sha256": sha256_file(snapshot), "max_wasm_stack": max_wasm_stack,
         "wasmtime": version("wasmtime")}, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run(out: Path, *, snapshot_file: Optional[Path] = None, contract_path: Path = CONTRACT_PATH,
        compile_: bool = True) -> str:
    contract: Dict[str, Any] = json.loads(contract_path.read_text(encoding="utf-8"))
    snap = check_contract_bundle(contract)["snapshot"]
    out.mkdir(parents=True, exist_ok=True)
    target = out / "snapshot.wasm"
    if snapshot_file is not None:
        _verify(snapshot_file, snap["sha256"], int(snap["size"]))
        if snapshot_file.resolve() != target.resolve():
            shutil.copyfile(snapshot_file, target)
    elif not (target.is_file() and sha256_file(target) == snap["sha256"]):
        tmp = target.with_suffix(".wasm.part")
        with urllib.request.urlopen(snap["url"], timeout=600) as response, tmp.open("wb") as fh:  # noqa: S310
            shutil.copyfileobj(response, fh)
        try:
            _verify(tmp, snap["sha256"], int(snap["size"]))
        except BundleError:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(target)
    _verify(target, snap["sha256"], int(snap["size"]))
    if compile_:
        precompile(target, out, int(contract["limits_floor"]["wasm_stack_bytes"]))
    return str(snap["sha256"])


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, default=None, help="use this file instead of downloading")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    try:
        digest = run(args.out, snapshot_file=args.snapshot, contract_path=args.contract,
                     compile_=not args.no_compile)
    except BundleError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"snapshot {digest} verified in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
