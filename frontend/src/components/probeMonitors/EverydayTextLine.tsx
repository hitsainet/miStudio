/**
 * The false-alarm checks rung 2 needs, in plain words: everyday text (ENH-006 D82) and nonsense text (D84).
 * "On everyday text: 0 false alarms in 1,200 (allowed up to 24)" / "Fired on 100% of nonsense text (2,000 rows)".
 *
 * Every sentence and number is the server's (`recompute_rung` → `everyday_text` / `nonsense_text` on the
 * report); this component only lays them out, so the panel and the MCP report cannot disagree.
 */
import type { EverydayTextCheck } from '../../types/probeMonitor';

interface TextCheckLineProps {
  check: EverydayTextCheck | null;
  /** "Everyday text" / "Nonsense text" — the label before a never-checked reason. */
  label: string;
  /** What staying quiet means for this kind of text, after a never-checked reason. */
  needs: string;
  testId: string;
}

export function TextCheckLine({ check, label, needs, testId }: TextCheckLineProps) {
  if (!check) return null;
  const tone = check.passed ? 'text-slate-400' : 'text-amber-300';
  return (
    <div className="mt-2 text-xs" data-testid={testId}>
      {check.sets.length === 0 ? (
        <p className={tone}>
          {label}: {check.reason}. {needs} (at least {check.min_rows.toLocaleString()} rows).
        </p>
      ) : (
        <>
          {check.sets.map((set) => (
            <p
              key={set.name}
              className={set.counts && set.within_allowance ? 'text-slate-400' : 'text-amber-300'}
              data-testid={`${testId}-set`}
            >
              {set.sentence}
            </p>
          ))}
          {!check.passed ? <p className={tone}>Rung 2 not reached: {check.reason}</p> : null}
        </>
      )}
    </div>
  );
}

export function EverydayTextLine({ check }: { check: EverydayTextCheck | null }) {
  return (
    <TextCheckLine
      check={check}
      label="Everyday text"
      needs="Rung 2 needs a probe to stay quiet on ordinary prose at its shipped bar"
      testId="everyday-text"
    />
  );
}

export function NonsenseTextLine({ check }: { check: EverydayTextCheck | null }) {
  return (
    <TextCheckLine
      check={check}
      label="Nonsense text"
      needs="Rung 2 needs a probe to stay quiet on text with no meaning (word salad, random characters, repeated tokens, garbled code) at its shipped bar"
      testId="nonsense-text"
    />
  );
}
