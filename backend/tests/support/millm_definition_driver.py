"""Validate exported probe definitions with miLLM's OWN contract model (`millm.api.schemas.probe.ProbeDefinitionV1`,
the model miLLM's import path validates with). Executed by miStudio's test in miLLM's venv, cwd = the miLLM
checkout, as a subprocess: nothing of miLLM is imported into miStudio. argv[1] is miServer's package name
(`millm`, or `miserver` after its ENH-009 P8 rename); the rest are definition file paths. Prints
{path: "ok" | the validation error}. Run without the package argument it exits 2 and never guesses."""
import importlib
import json
import sys

if len(sys.argv) < 2 or sys.argv[1] not in ("millm", "miserver"):
    sys.stderr.write("usage: millm_definition_driver <millm|miserver> <definition file>... (042 FR-042.36)\n")
    raise SystemExit(2)
ProbeDefinitionV1 = importlib.import_module(f"{sys.argv[1]}.api.schemas.probe").ProbeDefinitionV1

out = {}
for path in sys.argv[2:]:
    try:
        ProbeDefinitionV1.model_validate(json.loads(open(path, "rb").read()))
        out[path] = "ok"
    except Exception as exc:  # noqa: BLE001 - reported per file
        out[path] = f"{type(exc).__name__}: {exc}"[:500]
print(json.dumps(out))
