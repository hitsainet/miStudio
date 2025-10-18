/**
 * TimeRangeSelector Component
 *
 * Toggle buttons for selecting time range (1h, 6h, 24h)
 */

import { TimeRange } from '../../hooks/useHistoricalData';

interface TimeRangeSelectorProps {
  selected: TimeRange;
  onChange: (range: TimeRange) => void;
}

export function TimeRangeSelector({ selected, onChange }: TimeRangeSelectorProps) {
  const ranges: TimeRange[] = ['1h', '6h', '24h'];

  return (
    <div className="inline-flex rounded-lg border border-slate-700 bg-slate-900 p-1">
      {ranges.map((range) => (
        <button
          key={range}
          onClick={() => onChange(range)}
          className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
            selected === range
              ? 'bg-emerald-600 text-white'
              : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800'
          }`}
        >
          {range}
        </button>
      ))}
    </div>
  );
}
