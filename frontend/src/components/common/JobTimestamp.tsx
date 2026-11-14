/**
 * A job/run timestamp: date and time in US Eastern (see `formatJobTimestamp`), with the raw ISO
 * value on hover so an operator can correlate it with a UTC log line.
 */
import { formatJobTimestamp } from '../../utils/formatters';

interface JobTimestampProps {
  value: string | null | undefined;
  /** What to render when the value is absent or unparseable. */
  fallback?: string;
  className?: string;
  testId?: string;
}

export function JobTimestamp({ value, fallback = '—', className, testId }: JobTimestampProps) {
  const text = formatJobTimestamp(value);
  if (text === null || !value) {
    return <span className={className} data-testid={testId}>{fallback}</span>;
  }
  return (
    <time dateTime={value} title={value} className={className} data-testid={testId}>
      {text}
    </time>
  );
}
