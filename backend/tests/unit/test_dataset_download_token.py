"""A stored HuggingFace token must apply to dataset downloads too.

WHY THIS FILE EXISTS. `resolve_hf_token` centralises token precedence —
explicit request, then environment, then the encrypted `hf_token` setting — and
was wired into the SAE download path and J-lens acquisition. The DATASET
download path was never wired to it: it passed the request's `access_token`
straight to `load_dataset` and consulted nothing else.

So an operator who stored a token under Settings -> API Keys had it work for
SAEs and silently do nothing for gated datasets, failing with a 401 that reads
like a bad token rather than an absent one. Worse for an agent: I told the user
storing it there would unblock `lmsys/lmsys-chat-1m`, which was false.

These assert the CALL and its argument, because "reads a token from somewhere"
is not the claim — the claim is that this specific resolver, with its specific
precedence, is the one the download uses.
"""

import ast
import inspect

import pytest


def _download_source():
    from src.workers import dataset_tasks

    return inspect.getsource(dataset_tasks)


def _download_fn_node():
    tree = ast.parse(_download_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "download_dataset_task":
            return node
    raise AssertionError("download_dataset_task not found")


class TestTheDownloadResolvesTheStoredToken:

    def test_it_calls_resolve_hf_token(self):
        node = _download_fn_node()
        called = {
            sub.func.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        assert "resolve_hf_token" in called, (
            "a token stored in Settings must reach a dataset download, as it "
            "already does for SAEs and J-lens artifacts"
        )

    def test_the_request_token_is_what_gets_resolved(self):
        """Precedence starts with the request. Calling `resolve_hf_token()` with
        no argument would silently ignore the per-download Access Token field
        that the UI presents — the field an operator reaches for first."""
        node = _download_fn_node()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "resolve_hf_token"
            ):
                assert len(sub.args) == 1, "pass the request's access_token"
                assert isinstance(sub.args[0], ast.Name)
                assert sub.args[0].id == "access_token"
                return
        raise AssertionError("no resolve_hf_token call found")

    def test_load_dataset_receives_the_resolved_token_not_the_raw_one(self):
        """NEGATIVE CONTROL for the two above. Resolving a token and then
        handing `load_dataset` the raw `access_token` anyway would satisfy both
        — the resolver would be called and its result discarded, which is this
        arc's single most repeated defect."""
        node = _download_fn_node()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "load_dataset"
            ):
                tokens = [kw for kw in sub.keywords if kw.arg == "token"]
                assert tokens, "load_dataset must be given a token argument"
                value = tokens[0].value
                assert isinstance(value, ast.Name), "token must be a plain name"
                assert value.id == "resolved_token", (
                    f"load_dataset got `{getattr(value, 'id', value)}`; the "
                    "resolver's result is the only value that carries the "
                    "stored setting"
                )
                return
        raise AssertionError("no load_dataset call found in download_dataset_task")


class TestResolverPrecedence:
    """The resolver's own contract, which the download now depends on."""

    def test_an_explicit_token_wins(self, monkeypatch):
        from src.services import huggingface_sae_service as svc

        monkeypatch.setattr(svc.settings, "hf_token", "hf_from_env", raising=False)
        assert svc.resolve_hf_token("hf_from_request") == "hf_from_request"

    @pytest.mark.parametrize("empty", [None, "", "   ", "none", "None"])
    def test_a_missing_explicit_token_falls_through(self, monkeypatch, empty):
        """`"none"` is in here because it is what a UI sends when a user clears
        an optional field; passing it to HfApi yields a 401 that looks like a
        bad credential."""
        from src.services import huggingface_sae_service as svc

        monkeypatch.setattr(svc.settings, "hf_token", "hf_from_env", raising=False)
        assert svc.resolve_hf_token(empty) == "hf_from_env"

    def test_nothing_anywhere_is_none_not_empty_string(self, monkeypatch):
        """HfApi(token=None) is anonymous; HfApi(token="") is malformed."""
        from src.services import huggingface_sae_service as svc

        monkeypatch.setattr(svc.settings, "hf_token", None, raising=False)
        monkeypatch.setattr(
            svc, "SyncSessionLocal", None, raising=False
        )  # force the except branch
        assert svc.resolve_hf_token(None) is None
