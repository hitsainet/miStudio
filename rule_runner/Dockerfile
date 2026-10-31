# hitsai/mistudio-rule-runner (Feature 040 S4; FTDD Appendix C.4; spec D34, D41). Built in CI on the mirror only.
#
# miStudio's OWN runner host for operator code rules: a tiny HTTP server that runs each code rule inside the
# PUBLISHED WebAssembly guest of the vendored runner contract (miDataworks 018) on Wasmtime — a fresh instance per
# (text, function), fuel, a memory cap and an epoch deadline, through its own WASI table (no preopened directory,
# an empty environment, a fixed clock). It carries NO database credentials, no Redis URL, no data volume and no
# secret (Property 1, backend/tests/unit/test_rule_runner_manifest.py), and its NetworkPolicy denies egress
# (Property 2). A separate Deployment (`mistudio-rule-runner`), never a container of the GPU-visible, root,
# secret-holding backend pod (040-A1).
#
# It copies ONLY backend/src/rule_runner/ (installed as the top-level package `mistudio_rule_runner`; it imports
# nothing else from src — backend/tests/unit/test_rule_runner_is_self_contained.py) and the vendored contract.
# The guest snapshot is fetched at build time by the URL and sha256 READ FROM THE VENDORED CONTRACT (no literal
# digest here; `fetch` refuses a mismatch), then precompiled for this image's Wasmtime and CPU. No .wasm/.cwasm is
# ever committed (test_no_wasm_is_tracked).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=void
RUN useradd --create-home --uid 10001 runner
WORKDIR /app

COPY rule_runner/requirements.txt rule_runner/constraints.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt

COPY LICENSE /app/LICENSE
COPY backend/src/rule_runner/ /app/mistudio_rule_runner/
COPY backend/src/schemas/vendored/midataworks-runner-v1.json /app/schemas/vendored/midataworks-runner-v1.json

# Fetch the published guest snapshot by digest (from the vendored contract) and precompile it.
ENV RUNNER_BUNDLE_DIR=/app/bundle RUNNER_CONTRACT_PATH=/app/schemas/vendored/midataworks-runner-v1.json
RUN python -m mistudio_rule_runner.fetch --out "$RUNNER_BUNDLE_DIR" \
    && chown -R root:root /app && chmod -R a+rX /app

USER 10001
EXPOSE 8080
CMD ["python", "-m", "mistudio_rule_runner.server"]
