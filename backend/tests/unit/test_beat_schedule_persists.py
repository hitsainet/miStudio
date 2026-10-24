"""Celery beat keeps its schedule on the persistent volume.

The schedule file is where beat records when each periodic task last ran. It
lived in /tmp, which a pod restart empties, so beat counted every interval from
its own start: the daily checkpoint-retention task (`prune-checkpoints`) never
ran on a pod redeployed more than once a day, and the hourly ones (including the
spot-check excerpt expiry) only ran when a pod stayed up for an hour. Found
2026-10-21 checking tracker item 4.24: no prune run in 72 hours of logs while
pruning was enabled.

The entrypoint is EXECUTED with a fake `su` that records the command, as in
test_gpu_workers_entrypoint.py, rather than scraped.
"""

from pathlib import Path

from tests.unit.test_gpu_workers_entrypoint import _containers, _start
import yaml

BACKEND = Path(__file__).resolve().parents[2]
MANIFEST = BACKEND.parent / "k8s" / "base" / "backend.yaml"


def test_beat_writes_its_schedule_under_the_data_dir(tmp_path):
    launched = _start("celery-beat", tmp_path)
    data = tmp_path / "data"
    assert f"--schedule={data}/celerybeat/celerybeat-schedule" in launched, launched
    assert (data / "celerybeat").is_dir(), "the entrypoint must create the schedule directory"


def test_the_beat_container_mounts_the_persistent_data_volume():
    for doc in yaml.safe_load_all(MANIFEST.read_text()):
        if doc and doc.get("kind") == "Deployment":
            spec = doc["spec"]["template"]["spec"]
            for c in spec["containers"]:
                env = {e["name"]: e.get("value") for e in c.get("env", [])}
                if env.get("SERVICE_TYPE") != "celery-beat":
                    continue
                mounts = {m["mountPath"]: m["name"] for m in c.get("volumeMounts", [])}
                data_dir = env.get("DATA_DIR", "/data")
                assert data_dir in mounts, f"celery-beat does not mount {data_dir}: {mounts}"
                volumes = {v["name"]: v for v in spec.get("volumes", [])}
                vol = volumes[mounts[data_dir]]
                # hostPath (today) or a claim both outlive the pod; emptyDir does not.
                assert "persistentVolumeClaim" in vol or "hostPath" in vol, (
                    f"{data_dir} does not outlive the pod: {vol}")
                return
    raise AssertionError("no celery-beat container in the manifest")
