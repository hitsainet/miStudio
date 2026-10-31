"""miStudio's operator-code runner host (Feature 040 S4; FTDD Appendix C.4; spec D34, D41).

A SEPARATE PROCESS AND IMAGE (`hitsai/mistudio-rule-runner`, Deployment `mistudio-rule-runner`): it runs operator
code rules inside the published WebAssembly guest of the vendored runner contract
(`schemas/vendored/midataworks-runner-v1.json`, miDataworks 018) on Wasmtime — a fresh instance per (text, function),
three fuel budgets switched by phase markers accepted once and in order, a memory cap, an epoch backstop and the
host's OWN WASI table (no preopened directory, an empty environment, a fixed clock and random stream, no sockets, no
processes). The backend never imports Wasmtime and never runs code (FR-31); it calls this host over HTTP.

Written from the contract text (ADR-001: no import of miDataworks code), ported from the producer's host as miLLM
039 S3 ported it, keeping every producer Review R1 fix: bounds-checked guest memory, the `.cwasm` fallback, a bool
`repeat` refused, the lane backstop, the one-line reply, markers once and in order.

⚠ THIS PACKAGE IMPORTS NOTHING FROM THE APP. Only the standard library, ``wasmtime``, ``starlette``, ``uvicorn`` and
``anyio`` — and only RELATIVELY within itself, because the image installs it as a top-level package
(`mistudio_rule_runner`), without the rest of ``src`` (whose ``__init__`` imports SQLAlchemy).
`tests/unit/test_rule_runner_is_self_contained.py` walks every module's imports. ``contract`` alone is
standard-library only, so the backend may import it for the hash rule and the floor without pulling in Wasmtime.
"""
