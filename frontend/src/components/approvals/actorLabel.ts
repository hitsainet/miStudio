/**
 * Who did something, in plain words (ENH-009 042 FR-042.28, TI-16; FTID §3.15).
 *
 * This app's MCP server recorded itself as `agent:mistudio-mcp` before the rename and `agent:miworkbench-mcp`
 * after it. Both are the same agent: history rows keep the old id (DR-17, never migrated) and read the same as
 * new ones. Another `agent:<x>` is "an agent (<x>)"; anything else (an operator's name) is shown as stored.
 */
export const OWN_MCP_ACTORS: ReadonlySet<string> = new Set(['agent:mistudio-mcp', 'agent:miworkbench-mcp']);
export const OWN_MCP_LABEL = "an agent (miWorkbench's MCP server)";

export function actorLabel(actor: string | null | undefined): string {
  if (!actor) return '';
  if (OWN_MCP_ACTORS.has(actor)) return OWN_MCP_LABEL;
  if (actor.startsWith('agent:')) return `an agent (${actor.slice('agent:'.length)})`;
  return actor;
}

/** True when the stored actor is an agent (either own id, or any `agent:<x>`). */
export function isAgentActor(actor: string | null | undefined): boolean {
  return !!actor && actor.startsWith('agent:');
}
