/**
 * Both ids of this app's MCP server read as the same agent (ENH-009 042 FR-042.28, TI-16; FTASKS 12.23,
 * control K-12.42's frontend twin). History rows keep the old id (DR-17); new rows carry the new one.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { actorLabel, isAgentActor, OWN_MCP_LABEL } from './actorLabel';

describe('actorLabel', () => {
  it('both actor ids render as the miWorkbench MCP agent', () => {
    for (const id of ['agent:mistudio-mcp', 'agent:miworkbench-mcp']) {
      const { unmount } = render(<span title={id}>{actorLabel(id)}</span>);
      expect(screen.getByTitle(id)).toHaveTextContent(OWN_MCP_LABEL);
      expect(OWN_MCP_LABEL).toBe("an agent (miWorkbench's MCP server)");
      expect(isAgentActor(id)).toBe(true);
      unmount();
    }
  });

  it('another agent is named, and a person is shown as stored', () => {
    expect(actorLabel('agent:miforge-mcp')).toBe('an agent (miforge-mcp)');
    expect(actorLabel('sean')).toBe('sean');
    expect(isAgentActor('sean')).toBe(false);
    expect(actorLabel(null)).toBe('');
  });
});
