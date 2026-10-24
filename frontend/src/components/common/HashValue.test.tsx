import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import { HashValue, HASH_NOT_RECORDED } from './HashValue';

const DIGEST = 'ab'.repeat(32);

describe('HashValue (039 FR-20)', () => {
  it('shows the first 16 hex characters and the full value in the title', () => {
    render(<HashValue value={DIGEST} label="detector digest" />);
    const code = screen.getByTitle(DIGEST);
    expect(code.textContent).toBe(`${DIGEST.slice(0, 16)}…`);
  });

  it('copies the FULL value once, under an accessible name', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<HashValue value={DIGEST} label="decision digest" />);
    fireEvent.click(screen.getByRole('button', { name: 'Copy decision digest' }));
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText).toHaveBeenCalledWith(DIGEST);
  });

  it('says "not recorded" for a missing digest — never a placeholder hash', () => {
    render(<HashValue value={null} label="detector digest" testId="h" />);
    expect(screen.getByTestId('h').textContent).toBe(HASH_NOT_RECORDED);
    expect(screen.queryByRole('button')).toBeNull();
  });
});
