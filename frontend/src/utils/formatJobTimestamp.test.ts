/**
 * Job/run timestamps: date AND time, in US Eastern, year only when it is not the current one.
 *
 * The operator's report (2026-10-19): the training card header read "5:48:00 AM • 5h 8m" — a
 * clock time with no date. Every assertion below is on the exact string, so a regression to a
 * time-only or a viewer-zone format fails here regardless of the machine's own timezone.
 */
import { describe, it, expect } from 'vitest';
import { DISPLAY_TIME_ZONE, formatJobTimestamp } from './formatters';

const NOW = Date.parse('2026-10-19T16:00:00Z'); // Oct 19 2026, 12:00 PM EDT

describe('formatJobTimestamp', () => {
  it('formats in America/New_York', () => {
    expect(DISPLAY_TIME_ZONE).toBe('America/New_York');
  });

  it('shows the DATE as well as the time for a run that started today', () => {
    // The operator's own card: train_9e started 5:48 AM ET today.
    expect(formatJobTimestamp('2026-10-19T09:48:00Z', NOW)).toBe('Oct 19, 5:48 AM ET');
  });

  it('shows another day of the current year with its date and no year', () => {
    expect(formatJobTimestamp('2026-10-18T23:15:00Z', NOW)).toBe('Oct 18, 7:15 PM ET');
    expect(formatJobTimestamp('2026-02-03T17:05:00Z', NOW)).toBe('Feb 3, 12:05 PM ET');
  });

  it('includes the year for another year', () => {
    expect(formatJobTimestamp('2025-01-20T10:00:00Z', NOW)).toBe('Jan 20, 2025, 5:00 AM ET');
  });

  it('decides "today" and the year in Eastern time, not UTC', () => {
    // 01:30 UTC on Oct 20 is still Oct 19, 9:30 PM in New York.
    expect(formatJobTimestamp('2026-10-20T01:30:00Z', NOW)).toBe('Oct 19, 9:30 PM ET');
    // New Year's Eve in New York is already Jan 1 in UTC: still "this year", no year shown.
    const nye = Date.parse('2027-01-01T03:00:00Z'); // Dec 31 2026, 10 PM EST
    expect(formatJobTimestamp('2026-12-31T22:00:00Z', nye)).toBe('Dec 31, 5:00 PM ET');
    // ...while a 2026 stamp read from a 2027 Eastern day carries its year.
    const jan2 = Date.parse('2027-01-02T17:00:00Z');
    expect(formatJobTimestamp('2026-12-31T22:00:00Z', jan2)).toBe('Dec 31, 2026, 5:00 PM ET');
  });

  it('crosses the EDT/EST boundary on the right hour (Nov 1 2026, 2 AM EDT -> 1 AM EST)', () => {
    // Same UTC clock time, one day apart: UTC-4 before the change, UTC-5 after.
    expect(formatJobTimestamp('2026-10-31T09:30:00Z', NOW)).toBe('Oct 31, 5:30 AM ET');
    expect(formatJobTimestamp('2026-11-01T09:30:00Z', NOW)).toBe('Nov 1, 4:30 AM ET');
    // Either side of the instant itself: 05:59 UTC = 1:59 AM EDT, 06:00 UTC = 1:00 AM EST.
    expect(formatJobTimestamp('2026-11-01T05:59:00Z', NOW)).toBe('Nov 1, 1:59 AM ET');
    expect(formatJobTimestamp('2026-11-01T06:00:00Z', NOW)).toBe('Nov 1, 1:00 AM ET');
  });

  it('crosses the EST/EDT boundary in spring (Mar 8 2026, 2 AM EST -> 3 AM EDT)', () => {
    expect(formatJobTimestamp('2026-03-08T06:59:00Z', NOW)).toBe('Mar 8, 1:59 AM ET');
    expect(formatJobTimestamp('2026-03-08T07:00:00Z', NOW)).toBe('Mar 8, 3:00 AM ET');
  });

  it('accepts offset ISO strings, epoch millis and Date objects', () => {
    expect(formatJobTimestamp('2026-10-19T05:48:00-04:00', NOW)).toBe('Oct 19, 5:48 AM ET');
    expect(formatJobTimestamp(Date.parse('2026-10-19T09:48:00Z'), NOW)).toBe('Oct 19, 5:48 AM ET');
    expect(formatJobTimestamp(new Date('2026-10-19T09:48:00Z'), new Date(NOW))).toBe(
      'Oct 19, 5:48 AM ET',
    );
  });

  it('formats noon and midnight as 12 PM / 12 AM', () => {
    expect(formatJobTimestamp('2026-10-19T04:00:00Z', NOW)).toBe('Oct 19, 12:00 AM ET');
    expect(formatJobTimestamp('2026-10-19T16:00:00Z', NOW)).toBe('Oct 19, 12:00 PM ET');
  });

  it('returns null, never "Invalid Date", for an absent or unparseable value', () => {
    for (const bad of [null, undefined, '', 'not a date']) {
      expect(formatJobTimestamp(bad, NOW)).toBeNull();
    }
  });
});
