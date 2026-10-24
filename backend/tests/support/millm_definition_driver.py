"""Validate exported probe definitions with miLLM's OWN contract model (`millm.api.schemas.probe.ProbeDefinitionV1`,
the model miLLM's import path validates with). Executed by miStudio's test in miLLM's venv, cwd = the miLLM
checkout, as a subprocess: nothing of miLLM is imported into miStudio. argv: definition file paths. Prints
{path: "ok" | the validation error}."""
import json
import sys

from millm.api.schemas.probe import ProbeDefinitionV1

out = {}
for path in sys.argv[1:]:
    try:
        ProbeDefinitionV1.model_validate(json.loads(open(path, "rb").read()))
        out[path] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported per file
        out[path] = f"{type(exc).__name__}: {exc}"[:500]
print(json.dumps(out))
