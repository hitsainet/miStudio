import { describe, expect, it } from 'vitest';
import { formatExpiresIn, formatLocal, UNKNOWN_TIME } from './formatLocal';

describe('formatLocal (ENH-008 §2.7)', () => {
  it('constructs a real formatter without throwing, in any zone', () => {
    // The spec's literal dateStyle + timeZoneName options throw a TypeError; this must not.
    expect(() => formatLocal('2026-10-28T18:30:00Z')).not.toThrow();
    expect(() => formatLocal('2026-10-28T18:30:00Z', 'America/New_York')).not.toThrow();
  });

  it('shows the local time with its zone, and the exact UTC instant as the title', () => {
    const out = formatLocal('2026-10-28T18:30:00Z', 'America/New_York');
    expect(out.text).toMatch(/2026/);
    expect(out.text).toMatch(/2:30/);
    expect(out.text).toMatch(/EDT|GMT-4/);
    expect(out.title).toBe('2026-10-28T18:30:00.000Z');
  });

  it('says "unknown" for a missing or bad value, never "Invalid Date"', () => {
    for (const bad of [null, undefined, '', 'not a date']) {
      expect(formatLocal(bad)).toEqual({ text: UNKNOWN_TIME, title: '' });
    }
  });

  it('says how long until expiry', () => {
    const now = Date.parse('2026-10-28T12:00:00Z');
    expect(formatExpiresIn(null, now)).toBe('never expires');
    expect(formatExpiresIn('2026-10-29T11:00:00Z', now)).toBe('expires in 23 h');
    expect(formatExpiresIn('2026-10-28T12:40:00Z', now)).toBe('expires in 40 min');
    expect(formatExpiresIn('2026-10-28T11:00:00Z', now)).toBe('expired');
    expect(formatExpiresIn('junk', now)).toBe('expires unknown');
  });
});
