"""A circuit definition must be SERVABLE and its rung must be CHECKABLE.

Both defects here were found by running the real pipeline and then reading the
exported document — not by any test or audit.

DEFECT 1 — the export could not name its own SAEs.
`create_circuit(saes=[{"layer": 12, "sae_id": "sae_f2397db47ab9"}])` produced
`mistudio_sae_id: null`. `DefinitionSAERef` uses the wire name
`mistudio_sae_id`, and Pydantic's DEFAULT `extra="ignore"` silently discarded
`sae_id`. The definition validated, persisted and exported clean; it would only
fail much later at miLLM, as an unbound SAE, with nothing pointing back here.

Every other tool in this codebase calls that value `sae_id`
(`start_circuit_capture` takes `[{layer, sae_id}]`), so the wrong key is the
NATURAL thing to send.

DEFECT 2 — the export kept the rung and lost the evidence.
`CircuitService.from_candidates` populates each edge with its coactivation
statistics, attribution score, rung-2 effect size and validation_manifest_ref.
`create` stores exactly what it is handed. The evidence-preserving path is
`POST /circuit-discovery/{run_id}/build-circuit` — which had NO MCP TOOL, so an
agent could only reach `create_circuit` and could only produce evidence-free
circuits. A consumer got "trust me, it is rung 2" with no way to check.
"""

import inspect

import pytest


class TestTheDefinitionCanNameItsSAEs:
    def test_sae_id_is_accepted_as_an_alias(self):
        """The name every other tool uses must work, or the natural call is
        the broken one."""
        from src.schemas.cluster_profile import DefinitionSAERef

        ref = DefinitionSAERef(**{"layer": 12, "sae_id": "sae_f2397db47ab9"})
        assert ref.mistudio_sae_id == "sae_f2397db47ab9", (
            "`sae_id` was dropped — a circuit built this way exports with a "
            "null SAE id and cannot be served"
        )

    def test_the_wire_name_still_works(self):
        """Round-tripping an exported document must not regress."""
        from src.schemas.cluster_profile import DefinitionSAERef

        ref = DefinitionSAERef(
            **{"layer": 12, "mistudio_sae_id": "sae_f2397db47ab9"}
        )
        assert ref.mistudio_sae_id == "sae_f2397db47ab9"

    def test_the_published_schema_still_names_the_WIRE_field(self):
        """Guards the regression this fix nearly caused.

        A plain `alias="sae_id"` also renames the field on SERIALISATION. The
        republished JSON Schema then advertised `sae_id` and DROPPED
        `mistudio_sae_id` — and with extra="forbid", every previously exported
        document became INVALID. The schema-sync guard caught it; this pins it.
        """
        from src.schemas.cluster_profile import DefinitionSAERef

        props = DefinitionSAERef.model_json_schema()["properties"]
        assert "mistudio_sae_id" in props, (
            "the published schema no longer names `mistudio_sae_id` — every "
            "existing exported definition would fail validation"
        )

    def test_an_sae_entry_with_no_id_is_REJECTED_by_the_service(self):
        """The typo protection, at the layer that can express it.

        The contract model cannot use extra="forbid": pydantic omits validation
        aliases from the JSON Schema in both modes, so forbidding extras would
        publish a contract that rejects the `sae_id` the model accepts. The
        check therefore lives in CircuitService._validate, which can name both
        the offending key and its layer.
        """
        from src.services.circuit_service import (
            CircuitService, CircuitValidationError)

        with pytest.raises(CircuitValidationError) as exc:
            CircuitService._validate(
                name="t", narrative=None,
                saes=[{"layer": 12, "sae_idd": "typo"}],
                members=[{"layer": 12, "member_kind": "feature_ref",
                          "feature": {"feature_idx": 1, "strength": 1.0}}],
                edges=[], budget=None)
        msg = str(exc.value)
        assert "sae_idd" in msg and "layer 12" in msg, (
            f"the error must name the offending key and layer; got: {msg}"
        )

    def test_a_NULL_sae_id_is_REJECTED_and_blamed_on_the_manifest(self):
        """The evidence-preserving path can produce this shape itself.

        `from_candidates` builds each entry as
        `sae_by_layer.get(L, {}).get("sae_id")`. When a member's layer is not
        covered by the capture manifest that yields None — an unbound SAE that
        exports clean and fails at miLLM. The key IS present here, so the
        "you misspelled it" message would be actively misleading.
        """
        from src.services.circuit_service import (
            CircuitService, CircuitValidationError)

        with pytest.raises(CircuitValidationError) as exc:
            CircuitService._validate(
                name="t", narrative=None,
                saes=[{"mistudio_sae_id": None, "layer": 12}],
                members=[{"layer": 12, "member_kind": "feature_ref",
                          "feature": {"feature_idx": 1, "strength": 1.0}}],
                edges=[], budget=None)
        msg = str(exc.value)
        assert "null SAE id" in msg and "capture manifest" in msg, (
            f"the error must point at the manifest, not a typo; got: {msg}"
        )

    def test_a_valid_sae_entry_passes_the_service_guard(self):
        """Specificity: the guard must not reject correct input."""
        from src.services.circuit_service import CircuitService

        defn = CircuitService._validate(
            name="t", narrative=None,
            saes=[{"layer": 12, "sae_id": "sae_abc"}],
            members=[{"layer": 12, "member_kind": "feature_ref",
                      "feature": {"feature_idx": 1, "strength": 1.0}}],
            edges=[], budget=None)
        assert defn.saes[0].mistudio_sae_id == "sae_abc"

    def test_a_full_definition_round_trips_the_sae_id(self):
        """End to end through the contract model, not just the leaf."""
        from src.schemas.circuit_definition import CircuitDefinitionV1

        defn = CircuitDefinitionV1(
            name="t",
            saes=[{"layer": 12, "sae_id": "sae_abc"}],
            members=[{
                "layer": 12, "member_kind": "feature_ref",
                "feature": {"feature_idx": 1, "strength": 1.0},
            }],
        )
        assert defn.saes[0].mistudio_sae_id == "sae_abc"
        assert defn.model_dump(mode="json")["saes"][0]["mistudio_sae_id"] == "sae_abc"


class TestTheEvidencePreservingPathIsREACHABLE:
    """The rung is a claim; the evidence is what makes it checkable."""

    def _tools(self):
        import asyncio
        import os

        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        os.environ.setdefault("MILLM_API_URL", "http://millm.test")
        mcp, _c = build_server(
            MCPSettings(tool_categories="circuits", allow_anonymous=True),
            stdio=True,
        )
        return {t.name: t for t in asyncio.run(mcp.list_tools())}

    def test_build_circuit_from_discovery_is_registered(self):
        """It existed as a REST route with no MCP tool, so an agent could only
        reach create_circuit — and could only produce evidence-free circuits."""
        assert "build_circuit_from_discovery" in self._tools(), (
            "the evidence-preserving discovery->circuit path is unreachable "
            "from MCP; agents can only hand-assemble, which drops coactivation, "
            "attribution, effect_size and validation_manifest_ref"
        )

    def test_it_calls_the_build_endpoint_not_plain_create(self):
        """A tool that registers but posts to /circuits would look right and
        silently lose the evidence — the exact failure this closes."""
        from src.mcp_server.tools import circuits as mod

        src = inspect.getsource(mod.register)
        idx = src.index("async def build_circuit_from_discovery")
        body = src[idx : idx + 2000]
        assert "/build-circuit" in body, (
            "build_circuit_from_discovery does not post to the build-circuit "
            "endpoint, so it cannot be carrying discovery evidence"
        )

    def test_from_candidates_actually_populates_edge_evidence(self):
        """Guards the service the tool depends on. If this stops attaching
        evidence, the tool becomes a slower create_circuit."""
        from src.services.circuit_service import CircuitService

        src = inspect.getsource(CircuitService.from_candidates)
        for field in ("coactivation", "attribution", "effect_size",
                      "validation_manifest_ref"):
            assert field in src, (
                f"from_candidates no longer carries {field} onto its edges — "
                "the exported rung would lose its evidence"
            )

    def test_update_circuit_EXPOSES_saes(self):
        """The only field that makes an export servable must be settable.

        The PATCH route accepted `saes` all along
        (`endpoints/circuits.py` lists it in CircuitUpdate and in the
        cannot-be-nulled guard), but the MCP tool omitted the parameter — so a
        circuit persisted with a null SAE id could not be repaired by any
        agent. Same class as the unregistered tool: the capability existed and
        was unreachable.
        """
        tool = self._tools()["update_circuit"]
        props = (tool.inputSchema or {}).get("properties", {})
        assert "saes" in props, (
            "update_circuit does not expose `saes`, so a circuit exported "
            "with `mistudio_sae_id: null` cannot be repaired from MCP and "
            "can never be served through miLLM"
        )

    def test_update_circuit_actually_FORWARDS_saes(self):
        """Exposing the parameter and dropping it on the floor looks identical
        from the schema. Assert it reaches the request body."""
        from src.mcp_server.tools import circuits as mod

        src = inspect.getsource(mod.register)
        idx = src.index("async def update_circuit")
        body = src[idx : idx + 2500]
        assert '"saes": saes' in body, (
            "update_circuit accepts `saes` but never puts it in the PATCH "
            "body — the repair would silently no-op"
        )

    def test_create_circuit_STEERS_callers_to_the_better_path(self):
        """Docstrings are the only guidance an agent gets before choosing.

        create_circuit is still correct for hand assembly; it must just not be
        the default choice for a discovered circuit.
        """
        desc = self._tools()["create_circuit"].description or ""
        assert "build_circuit_from_discovery" in desc, (
            "create_circuit does not point at the evidence-preserving path, so "
            "an agent has no reason to prefer it"
        )
