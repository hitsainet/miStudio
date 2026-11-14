"""T-32 (042 FTASKS 13.4, 13.5; FTDD §11.3; FR-042.22, DR-28, DR-42; S001 FTDD §16 finding 7).

* Every `hostPath` (pod volumes and PersistentVolumes) is `type: Directory`: a missing folder refuses to mount
  (`FailedMount`, rehearsed in S001 2.08) instead of starting an empty database on a fresh folder.
* No PersistentVolume sits at the shared parent `/mnt/hdd2/pv`; this app's backup PV is its own folder
  `/mnt/hdd2/pv/miworkbench`, Retain, `local-hdd`, pinned to the node the pods are pinned to, and the claim binds
  it by `volumeName`.
* `BACKUP_DIR` equals the backup volume's `mountPath`, so dumps land at the PV root (where S001's restore drill and
  the moved dumps are), never in a nested `mistudio/` folder that `mkdir -p` would make silently.
* The prune globs match the file name the job writes (DR-42 keeps the `mistudio-` prefix).
"""
from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

from tests.support import k8s_render

BACKUP_PV = "miworkbench-db-backups-pv"
BACKUP_PATH = "/mnt/hdd2/pv/miworkbench"
HDD2_PARENT = "/mnt/hdd2/pv"


def _pvs(docs):
    return [d for d in docs if d.get("kind") == "PersistentVolume"]


def _pv_path(pv: dict) -> str:
    spec = pv["spec"]
    return posixpath.normpath((spec.get("hostPath") or spec.get("local") or {}).get("path", ""))


def _cron(docs):
    (cron,) = [d for d in docs if d.get("kind") == "CronJob"]
    return cron


def test_hostpaths_are_directory_typed():
    docs = k8s_render.render()
    found = {}
    for d in docs:
        for pod in k8s_render.pod_specs(d):
            for v in pod.get("volumes") or []:
                if "hostPath" in v:
                    found[f"{d['metadata']['name']}/{v['name']}"] = (v["hostPath"]["path"], v["hostPath"].get("type"))
        if d.get("kind") == "PersistentVolume" and "hostPath" in d["spec"]:
            found[f"pv/{d['metadata']['name']}"] = (d["spec"]["hostPath"]["path"], d["spec"]["hostPath"].get("type"))
    assert len(found) >= 4, found  # data, postgres, redis, the backup PV
    wrong = {k: v for k, v in found.items() if v[1] != "Directory"}
    assert not wrong, f"untyped or DirectoryOrCreate hostPath: {wrong}"
    assert {p for p, _ in found.values()} >= {"/data/miworkbench/data", "/data/miworkbench/postgres",
                                               "/data/miworkbench/redis", BACKUP_PATH}, found


def test_no_pv_at_hdd2_parent():
    docs = k8s_render.render()
    assert _pvs(docs), "no PersistentVolume rendered: the backup PV is missing"
    paths = {d["metadata"]["name"]: _pv_path(d) for d in _pvs(docs)}
    assert HDD2_PARENT not in paths.values(), paths
    for a, pa in paths.items():
        for b, pb in paths.items():
            assert a == b or not pb.startswith(pa.rstrip("/") + "/"), f"PV {a} ({pa}) is a parent of {b} ({pb})"


def test_backup_pv_is_own_folder():
    docs = k8s_render.render()
    (pv,) = [d for d in _pvs(docs) if d["metadata"]["name"] == BACKUP_PV]
    spec = pv["spec"]
    assert _pv_path(pv) == BACKUP_PATH, spec
    assert spec["persistentVolumeReclaimPolicy"] == "Retain"
    assert spec["storageClassName"] == "local-hdd"
    terms = spec["nodeAffinity"]["required"]["nodeSelectorTerms"]
    hosts = {v for t in terms for e in t["matchExpressions"] if e["key"] == "kubernetes.io/hostname"
             for v in e["values"]}
    pinned = {pod["nodeSelector"]["kubernetes.io/hostname"] for d in docs for pod in k8s_render.pod_specs(d)
              if (pod.get("nodeSelector") or {}).get("kubernetes.io/hostname")}
    assert pinned and hosts == pinned, (hosts, pinned)
    claims = [d for d in docs if d.get("kind") == "PersistentVolumeClaim"]
    cron_claims = {v["persistentVolumeClaim"]["claimName"] for pod in k8s_render.pod_specs(_cron(docs))
                   for v in pod.get("volumes") or [] if "persistentVolumeClaim" in v}
    (claim,) = [c for c in claims if c["metadata"]["name"] in cron_claims]
    assert claim["spec"]["volumeName"] == BACKUP_PV
    assert claim["spec"]["storageClassName"] == spec["storageClassName"]


def _backup_container(cron):
    (pod,) = k8s_render.pod_specs(cron)
    (c,) = pod["containers"]
    return pod, c


def test_backup_dir_equals_mount_path():
    docs = k8s_render.render()
    pod, c = _backup_container(_cron(docs))
    claim_volumes = {v["name"] for v in pod["volumes"] if "persistentVolumeClaim" in v}
    (mount,) = [m["mountPath"] for m in c["volumeMounts"] if m["name"] in claim_volumes]
    env = {e["name"]: e.get("value") for e in c["env"]}
    assert posixpath.normpath(env["BACKUP_DIR"]) == posixpath.normpath(mount), (env["BACKUP_DIR"], mount)


def test_prune_glob_matches_dump_name(tmp_path: Path):
    docs = k8s_render.render()
    _, c = _backup_container(_cron(docs))
    script = "\n".join(c.get("command", []) + c.get("args", []))
    code = [ln.strip() for ln in script.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    (target,) = [ln for ln in code if ln.startswith("target=")]
    globs = sorted({m for ln in code for m in re.findall(r'"\$BACKUP_DIR"/([^\s|)]+\.dump)', ln)})
    assert len(globs) == 1, globs  # the newest-dump read, the count, the prune and the retained line share one glob
    made = subprocess.run(["bash", "-c", f'BACKUP_DIR="{tmp_path}"; stamp=20261110T033000Z; {target}; '
                                         'touch "$target"; basename "$target"'],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "other-20261110T033000Z.dump").touch()
    listed = subprocess.run(["bash", "-c", f'BACKUP_DIR="{tmp_path}"; ls -1 "$BACKUP_DIR"/{globs[0]}'],
                            capture_output=True, text=True).stdout.split()
    assert [Path(p).name for p in listed] == [made], (globs, listed, made)
