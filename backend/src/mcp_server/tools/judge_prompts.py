"""Judge-prompt library tools (category: probe_monitors) — operator decisions D1/D2, PADR IDL-57.

Registered beside `probe_monitors` in `tools/__init__.CATEGORY_MODULES`, so the category's
deployment switches cover them; every route under ``/probe-monitors/judge-prompts`` (and the
probe-behaviour route) is called by exactly one tool here, which `test_mcp_route_parity` checks
against the live registry. Pure HTTP through the miStudio client (memory:
mcp-server-is-a-separate-deployment) — including the miDataworks rubric import, which miStudio's
backend fetches, so this pod needs no miDataworks client for it.

WHAT AN AGENT NEEDS TO KNOW, in the descriptions because that is what an agent reads:

* a judge prompt is a MEASURING INSTRUMENT: a version that a run has used is locked, and changing
  it means `new_judge_prompt_version`;
* rung 3 counts a judge run only when the prompt's ``behaviour`` equals the probe's — so making a
  prompt for a new behaviour is how a new kind of probe can reach rung 3 at all;
* the answer form (rating, yes/no, label set) decides how replies are read; check a new prompt
  with `preview_judge_prompt` (free) and `try_judge_prompt` (a few real judge calls) first.

No approval gate on these: library edits change no GPU state, and `try_judge_prompt` sends at
most 10 rows and never loads a model. The judge RUN is gated (`start_probe_judge_run`).
"""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..client import MiStudioClient
from ..config import MCPSettings

#: Recorded as `created_by` on what an agent makes.
AGENT_ACTOR = "agent:mistudio-mcp"

ANSWER_FORM_HELP = (
    "How the reply is read, one of: "
    '{"kind": "rating", "min": 1, "max": 10, "positive_from": 6, "json_key": "rating"} '
    '(a whole number; positive_from and above = behaviour present); '
    '{"kind": "binary", "yes_token": "yes", "no_token": "no", "positive_answer": "yes", '
    '"json_key": "answer"}; '
    '{"kind": "labels", "labels": ["a", "b", "c"], "positive_labels": ["a"], "json_key": "label"}. '
    'Optional on each: "reply": "json" (default; a JSON object with json_key) or "verdict_line" '
    '(last line "VERDICT: <value>"); binary/labels also take "case_sensitive". A reply that '
    "cannot be read counts toward the run's parse-failure limit — it is never guessed"
)
TEMPLATE_HELP = (
    "The judge's user message. It must contain at least one placeholder, and only these: "
    "{interaction} (the conversation as 'role: text' lines), {text} (the turns' text without "
    "roles), {last_user_message}, {last_assistant_message}. Long rows are cut from the front"
)


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def register(mcp: MCPServer, client: MiStudioClient, settings: MCPSettings) -> None:  # noqa: ARG001
    @mcp.tool()
    async def list_judge_prompts(
        behaviour: Annotated[Optional[str], Field(description="Only prompts that judge this behaviour key (e.g. '<behaviour-key>')")] = None,
        name: Annotated[Optional[str], Field(description="Only the versions of this prompt name")] = None,
        include_archived: Annotated[bool, Field(description="Also list archived versions")] = False,
    ) -> Any:
        """List the judge-prompt library: each version's key (name/vN), behaviour, answer form,
        content hash, whether it is locked (used by a run) and by how many runs. Also returns the
        placeholders a template may use. Pick a version whose behaviour matches the probe's before
        starting a judge run."""
        return await client.get(
            "/probe-monitors/judge-prompts",
            behaviour=behaviour,
            name=name,
            include_archived=include_archived,
        )

    @mcp.tool()
    async def get_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
    ) -> Any:
        """One judge prompt version in full (instructions, template, answer form, hash), with every
        version of its name and how many runs used each."""
        return await client.get(f"/probe-monitors/judge-prompts/{prompt_id}")

    @mcp.tool()
    async def create_judge_prompt(
        name: Annotated[str, Field(description="New prompt name: lower-case words joined by hyphens, e.g. 'refusal-presence'")],
        title: Annotated[str, Field(description="What a person reads in a list, e.g. 'Refusal present'")],
        behaviour: Annotated[str, Field(description="The behaviour key this prompt judges (lower-case, hyphens). Rung 3 matches it against the probe's behaviour")],
        system_prompt: Annotated[str, Field(description="The judge's instructions: what counts as positive, and the exact reply format the answer form reads")],
        answer_form: Annotated[Dict[str, Any], Field(description=ANSWER_FORM_HELP)],
        user_template: Annotated[str, Field(description=TEMPLATE_HELP)] = "{interaction}",
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words")] = None,
        description: Annotated[Optional[str], Field(description="Longer notes for people choosing a prompt")] = None,
        max_interaction_chars: Annotated[int, Field(description="How many characters of each row the judge sees (200-100000)", ge=200, le=100000)] = 8000,
    ) -> Any:
        """Create version 1 of a NEW judge prompt — how a new classification domain gets a judge.
        Refused (409) when the name exists: use new_judge_prompt_version. Check it with
        preview_judge_prompt and try_judge_prompt before running it."""
        return await client.post(
            "/probe-monitors/judge-prompts",
            json_body=_drop_none(
                {
                    "name": name,
                    "title": title,
                    "behaviour": behaviour,
                    "behaviour_label": behaviour_label,
                    "description": description,
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "answer_form": answer_form,
                    "max_interaction_chars": max_interaction_chars,
                    "actor": AGENT_ACTOR,
                }
            ),
        )

    @mcp.tool()
    async def update_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
        title: Annotated[Optional[str], Field(description="New title")] = None,
        description: Annotated[Optional[str], Field(description="New description")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="New behaviour words")] = None,
        behaviour: Annotated[Optional[str], Field(description="New behaviour key — only while no run has used this version")] = None,
        system_prompt: Annotated[Optional[str], Field(description="New instructions — only while unused")] = None,
        user_template: Annotated[Optional[str], Field(description=TEMPLATE_HELP + " — only while unused")] = None,
        answer_form: Annotated[Optional[Dict[str, Any]], Field(description=ANSWER_FORM_HELP + " — only while unused")] = None,
        max_interaction_chars: Annotated[Optional[int], Field(description="Only while unused", ge=200, le=100000)] = None,
    ) -> Any:
        """Edit a version IN PLACE. Title, description and behaviour words always; the
        instructions, template, answer form and behaviour key only while no judge run has used it
        (409 prompt_locked otherwise — make a new version, so recorded runs keep their instrument)."""
        return await client.patch(
            f"/probe-monitors/judge-prompts/{prompt_id}",
            json_body=_drop_none(
                {
                    "title": title,
                    "description": description,
                    "behaviour_label": behaviour_label,
                    "behaviour": behaviour,
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "answer_form": answer_form,
                    "max_interaction_chars": max_interaction_chars,
                    "actor": AGENT_ACTOR,
                }
            ),
        )

    @mcp.tool()
    async def new_judge_prompt_version(
        prompt_id: Annotated[str, Field(description="The version to start from (jp_…)")],
        system_prompt: Annotated[Optional[str], Field(description="Changed instructions")] = None,
        user_template: Annotated[Optional[str], Field(description=TEMPLATE_HELP)] = None,
        answer_form: Annotated[Optional[Dict[str, Any]], Field(description=ANSWER_FORM_HELP)] = None,
        max_interaction_chars: Annotated[Optional[int], Field(description="Changed length (200-100000)", ge=200, le=100000)] = None,
        title: Annotated[Optional[str], Field(description="Title for the new version")] = None,
        description: Annotated[Optional[str], Field(description="Description for the new version")] = None,
    ) -> Any:
        """Make the NEXT version of a prompt (same name and behaviour) from an existing one plus
        changes. Refused when nothing that is sent would change. A different behaviour is a
        different prompt: use clone_judge_prompt."""
        return await client.post(
            f"/probe-monitors/judge-prompts/{prompt_id}/versions",
            json_body=_drop_none(
                {
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "answer_form": answer_form,
                    "max_interaction_chars": max_interaction_chars,
                    "title": title,
                    "description": description,
                    "actor": AGENT_ACTOR,
                }
            ),
        )

    @mcp.tool()
    async def clone_judge_prompt(
        prompt_id: Annotated[str, Field(description="The version to copy (jp_…)")],
        name: Annotated[str, Field(description="The new prompt's name (lower-case, hyphens)")],
        title: Annotated[Optional[str], Field(description="The new prompt's title")] = None,
        behaviour: Annotated[Optional[str], Field(description="The behaviour key it judges; omit to keep the original's")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words")] = None,
        system_prompt: Annotated[Optional[str], Field(description="Changed instructions")] = None,
        user_template: Annotated[Optional[str], Field(description=TEMPLATE_HELP)] = None,
        answer_form: Annotated[Optional[Dict[str, Any]], Field(description=ANSWER_FORM_HELP)] = None,
    ) -> Any:
        """Copy a version into a NEW prompt (version 1 under a new name) — the quick way to start
        a judge for a related behaviour."""
        return await client.post(
            f"/probe-monitors/judge-prompts/{prompt_id}/clone",
            json_body=_drop_none(
                {
                    "name": name,
                    "title": title,
                    "behaviour": behaviour,
                    "behaviour_label": behaviour_label,
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "answer_form": answer_form,
                    "actor": AGENT_ACTOR,
                }
            ),
        )

    @mcp.tool()
    async def archive_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
    ) -> Any:
        """Hide a version from new runs. Every run that used it keeps working and keeps its link."""
        return await client.post(f"/probe-monitors/judge-prompts/{prompt_id}/archive")

    @mcp.tool()
    async def unarchive_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
    ) -> Any:
        """Make an archived version available to new runs again."""
        return await client.post(f"/probe-monitors/judge-prompts/{prompt_id}/unarchive")

    @mcp.tool()
    async def delete_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
    ) -> Any:
        """Delete a version NO run has used (a draft). A used version is refused (409): it is the
        instrument of a recorded measurement — archive it instead."""
        return await client.delete(f"/probe-monitors/judge-prompts/{prompt_id}")

    @mcp.tool()
    async def preview_judge_prompt(
        dataset_id: Annotated[str, Field(description="A probe dataset view (pmd_…) whose rows to render")],
        prompt_id: Annotated[Optional[str], Field(description="A saved version (jp_…); or give draft instead")] = None,
        draft: Annotated[Optional[Dict[str, Any]], Field(description="An unsaved prompt: {system_prompt, user_template, answer_form, max_interaction_chars}")] = None,
        rows: Annotated[int, Field(description="How many rows (1-20), half of each label where possible", ge=1, le=20)] = 4,
    ) -> Any:
        """Show the judge's exact messages for a few rows of a view, plus warnings when the
        instructions do not ask for the shape the answer form reads. Calls NO model; free."""
        return await client.post(
            "/probe-monitors/judge-prompts/preview",
            json_body=_drop_none(
                {"dataset_id": dataset_id, "prompt_id": prompt_id, "draft": draft, "rows": rows}
            ),
        )

    @mcp.tool()
    async def try_judge_prompt(
        dataset_id: Annotated[str, Field(description="A probe dataset view (pmd_…) to sample rows from")],
        endpoint: Annotated[str, Field(description="OpenAI-compatible base URL of the judge")],
        model: Annotated[str, Field(description="The judge model name at that endpoint (must already be loaded)")],
        prompt_id: Annotated[Optional[str], Field(description="A saved version (jp_…); or give draft instead")] = None,
        draft: Annotated[Optional[Dict[str, Any]], Field(description="An unsaved prompt: {system_prompt, user_template, answer_form, max_interaction_chars}")] = None,
        rows: Annotated[int, Field(description="How many rows (1-10); each is one judge call", ge=1, le=10)] = 4,
    ) -> Any:
        """Send a few rows through the judge and return each reply, how it was read, and whether
        it agrees with the row's label. Records nothing, does not lock the prompt, and NEVER loads
        a model (a judge run does)."""
        return await client.post(
            "/probe-monitors/judge-prompts/try",
            json_body=_drop_none(
                {
                    "dataset_id": dataset_id,
                    "endpoint": endpoint,
                    "model": model,
                    "prompt_id": prompt_id,
                    "draft": draft,
                    "rows": rows,
                }
            ),
        )

    @mcp.tool()
    async def export_judge_prompt(
        prompt_id: Annotated[str, Field(description="Judge prompt version id (jp_…)")],
    ) -> Any:
        """A portable `mistudio.judge-prompt/v1` document of one version; importing it elsewhere
        reproduces the same name/vN and content hash."""
        return await client.get(f"/probe-monitors/judge-prompts/{prompt_id}/export")

    @mcp.tool()
    async def import_judge_prompt(
        document: Annotated[Dict[str, Any], Field(description="A mistudio.judge-prompt/v1 document from export_judge_prompt")],
    ) -> Any:
        """Store an exported version under its own name and version. The same document twice
        returns the stored row; a different document under a stored name/vN is refused (409);
        a document whose text does not match its hash is refused (422)."""
        return await client.post(
            "/probe-monitors/judge-prompts/import",
            json_body={"document": document, "actor": AGENT_ACTOR},
        )

    @mcp.tool()
    async def list_midataworks_rubrics_for_judging() -> Any:
        """miDataworks' rubrics, each marked importable or not (binary and pointwise only). 503
        when miDataworks is not connected to miStudio — then import an exported rubric document
        with import_judge_prompt_from_rubric(document=…) instead."""
        return await client.get("/probe-monitors/judge-prompts/midataworks/rubrics")

    @mcp.tool()
    async def import_judge_prompt_from_rubric(
        name: Annotated[str, Field(description="The new prompt's name (lower-case, hyphens)")],
        title: Annotated[str, Field(description="The new prompt's title")],
        behaviour: Annotated[str, Field(description="The behaviour key it judges")],
        rubric_id: Annotated[Optional[str], Field(description="A miDataworks rubric id (needs miDataworks connected); or give document")] = None,
        document: Annotated[Optional[Dict[str, Any]], Field(description="A midataworks.rubric/v1 export; works without miDataworks")] = None,
        field_map: Annotated[Optional[Dict[str, str]], Field(description="Rubric field → placeholder (interaction, text, last_user_message, last_assistant_message)")] = None,
        literals: Annotated[Optional[Dict[str, str]], Field(description="Rubric field → fixed text (e.g. its question)")] = None,
        positive_verdicts: Annotated[Optional[List[str]], Field(description="For a label rubric: the verdicts meaning the behaviour is present")] = None,
        positive_from: Annotated[Optional[int], Field(description="For a rating rubric: the lowest rating meaning present")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words")] = None,
    ) -> Any:
        """Turn a miDataworks rubric into a library prompt (version 1 of `name`). Binary and
        pointwise rubrics with one system and one user message map; pairwise and stepwise are
        refused with the reason. Every rubric field must be mapped or given a fixed value."""
        return await client.post(
            "/probe-monitors/judge-prompts/import-rubric",
            json_body=_drop_none(
                {
                    "name": name,
                    "title": title,
                    "behaviour": behaviour,
                    "behaviour_label": behaviour_label,
                    "rubric_id": rubric_id,
                    "document": document,
                    "field_map": field_map,
                    "literals": literals,
                    "positive_verdicts": positive_verdicts,
                    "positive_from": positive_from,
                    "actor": AGENT_ACTOR,
                }
            ),
        )

    @mcp.tool()
    async def set_probe_behaviour(
        probe_id: Annotated[str, Field(description="Probe id (pm_…)")],
        behaviour: Annotated[Optional[str], Field(description="What the probe detects, as a key (lower-case, hyphens); omit to clear")] = None,
        behaviour_label: Annotated[Optional[str], Field(description="The behaviour in words")] = None,
    ) -> Any:
        """Say what a probe detects, then its rung is recomputed: only judge runs whose prompt
        judges this same behaviour count toward rung 3. Clearing it removes that credit."""
        return await client.patch(
            f"/probe-monitors/probes/{probe_id}/behaviour",
            json_body={"behaviour": behaviour, "behaviour_label": behaviour_label},
        )
