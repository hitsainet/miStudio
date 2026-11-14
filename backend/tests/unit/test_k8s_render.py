"""042 FTASKS 13.1 (FTID §11.2, TI-20, TPD-18): the manifest tests' one rendering path, in both modes.

`kustomize build` (or `kubectl kustomize`) and the `resources:`-list fallback must give the same objects and the
same images, so a manifest test run where neither tool is installed reads what the cluster gets.
"""
from __future__ import annotations

import pytest
import yaml

from tests.support import k8s_render


def _summary(docs):
    return sorted((k8s_render.key(d), tuple(sorted(c["image"] for c in k8s_render.containers(d)))) for d in docs)


def test_kustomize_and_resource_list_render_the_same_set():
    if k8s_render.tool() is None:
        pytest.skip("KUSTOMIZE SKIPPED LOUDLY: neither kustomize nor kubectl is on PATH; the fallback alone ran")
    built = k8s_render.render_with_tool()
    listed = k8s_render.render_from_resource_list()
    assert _summary(built) == _summary(listed)
    assert len(built) >= 15


def test_resource_list_fallback(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "kustomization.yaml").write_text(yaml.safe_dump({
        "resources": ["a.yaml", "sub"], "images": [{"name": "hitsai/x", "newTag": "v2"}]}))
    (tmp_path / "a.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata: {name: a, namespace: n}\n"
        "spec: {template: {spec: {containers: [{name: c, image: 'hitsai/x:latest'}, {name: d, image: 'redis:7'}]}}}\n"
        "---\napiVersion: v1\nkind: Service\nmetadata: {name: a, namespace: n}\n")
    (tmp_path / "sub" / "kustomization.yaml").write_text(yaml.safe_dump({"resources": ["b.yaml"]}))
    (tmp_path / "sub" / "b.yaml").write_text("apiVersion: v1\nkind: Service\nmetadata: {name: b, namespace: m}\n")
    docs = k8s_render.render_from_resource_list(tmp_path)
    assert [k8s_render.key(d) for d in docs] == [("Deployment", "n", "a"), ("Service", "n", "a"), ("Service", "m", "b")]
    assert [c["image"] for c in k8s_render.containers(docs[0])] == ["hitsai/x:v2", "redis:7"]
    real = k8s_render.render_from_resource_list()
    assert {"Namespace", "Deployment", "Service", "Ingress", "CronJob"} <= {d["kind"] for d in real}
    if k8s_render.tool() is not None:
        assert _summary(k8s_render.render_with_tool(tmp_path)) == _summary(docs)
