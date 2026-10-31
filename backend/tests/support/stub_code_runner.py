"""A STUB of miStudio's runner host's HTTP surface for unit tests (Feature 040 S4) — never a runner of code.

`install(monkeypatch)` points `services.code_rule_runner` at an `httpx.MockTransport` that answers `/v1/run` and
`/v1/inspect` in the runner's wire shape, recording every request (payload and count) so wiring tests assert what
the backend SENT. It never reads or executes the document's source: the count a "function" returns is decided by
the call's PARAMS, which the tests write:

* ``{"needle": s}``          → the number of occurrences of ``s`` in the text;
* ``{"fail": reason}``       → ``{"reason": reason}`` for every text;
* ``{"fail_on": s, "reason": r}`` → the reason for a text containing ``s``, else the needle count (``needle`` param);
* anything else              → ``{"count": 0}``.

`documents()` builds `midataworks.runner/v1` code-rule documents with random function names; `code_signal()` the
rule-set signal that names one. The real runner over the published guest is exercised by `rule_runner/tests/`
and `tests/unit/test_code_rule_hostile.py` (CI's `rule-runner-tests` job).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.rule_runner import contract as runner_contract

URL = "http://stub-rule-runner:8080"


@dataclass
class StubRunner:
    requests: List[Dict[str, Any]] = field(default_factory=list)
    #: Called before answering each /v1/run (e.g. to revoke an approval between two chunks).
    before_run: Optional[Callable[[int], None]] = None
    #: An HTTP status to answer instead (a runner fault).
    status: Optional[int] = None
    #: Override the bundle digest the stub reports (a runner of another bundle).
    bundle_digest: Optional[str] = None
    #: For approval-request cases: answers per (case index) pair, overriding the params rule.
    pairs: Optional[List[List[Dict[str, Any]]]] = None

    def runs(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["path"] == "/v1/run"]

    @staticmethod
    def answer(params: Dict[str, Any], text: str) -> Dict[str, Any]:
        if "fail" in params:
            return {"reason": params["fail"]}
        if "fail_on" in params and params["fail_on"] in text:
            return {"reason": params["reason"]}
        if "needle" in params:
            return {"count": min(text.count(params["needle"]), 1000)}
        return {"count": 0}

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append({"path": request.url.path, "body": body})
        if self.status is not None:
            return httpx.Response(self.status, json={"error": {"code": "runner_internal"}})
        if request.url.path == "/v1/inspect":
            return httpx.Response(200, json={"functions": ["f"], "imports": []})
        if self.before_run is not None:
            self.before_run(len(self.runs()))
        identity = runner_contract.identity()
        results: List[List[Dict[str, Any]]] = []
        for i, call in enumerate(body["calls"]):
            if self.pairs is not None:
                results.append(self.pairs[i])
                continue
            out: List[Dict[str, Any]] = []
            for text in call["texts"]:
                for _ in range(int(call.get("repeat", 1))):
                    out.append(self.answer(call["params"], text))
            results.append(out)
        return httpx.Response(200, json={"contract_version": identity["contract_version"],
                                         "bundle_digest": self.bundle_digest or identity["bundle_digest"],
                                         "results": results})


def install(monkeypatch: Any, *, enabled: bool = True, **settings: Any) -> StubRunner:
    """Point the backend's runner client at a fresh stub; with ``enabled`` turn the runner on in the runtime
    settings (URL + `code_rules_enabled`), plus any other overrides given."""
    from src.core import runtime_settings
    from src.services import code_rule_runner

    stub = StubRunner()
    monkeypatch.setattr(code_rule_runner, "_transport_factory", lambda: httpx.MockTransport(stub.handle))
    monkeypatch.setattr(code_rule_runner, "RETRY_SLEEP_S", 0.0)
    overrides = {"code_rules_enabled": True} if enabled else {}
    overrides.update(settings)
    from src.core.config import settings as app_settings

    monkeypatch.setattr(app_settings, "code_rule_runner_url", URL if enabled else "")
    if overrides:
        runtime_settings.apply(overrides)
    return stub


def document(function: str = "f", *, source: str = "def f(text, params):\n    return 0\n",
             statement: str = "test fixture") -> Dict[str, Any]:
    return {"runner_contract": runner_contract.FORMAT, "source": source,
            "functions": [{"name": function, "output": {"type": "count"}}],
            "params_schema": {"type": "object"},
            "derived_from": {"sources": ["tests"], "statement": statement}}


def code_signal(sid: str, doc: Dict[str, Any], params: Optional[Dict[str, Any]] = None,
                function: Optional[str] = None) -> Dict[str, Any]:
    return {"id": sid, "kind": "code_rule", "code_sha256": runner_contract.code_sha256(doc),
            "runner_contract": runner_contract.FORMAT, "function": function or doc["functions"][0]["name"],
            "params": dict(params or {}), "outputs": [{"name": "hits", "type": "int"}]}
