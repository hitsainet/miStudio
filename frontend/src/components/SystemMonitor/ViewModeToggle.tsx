/**
 * ViewModeToggle Component
 *
 * Toggle between single GPU view and comparison view
 */

import { Monitor, Grid } from 'lucide-react';

export type ViewMode = 'single' | 'compare';

interface ViewModeToggleProps {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
  disabled?: boolean;
}

export function ViewModeToggle({ mode, onChange, disabled = false }: ViewModeToggleProps) {
  return (
    <div className="inline-flex rounded-lg border border-slate-700 bg-slate-900 p-1">
      <button
        onClick={() => onChange('single')}
        disabled={disabled}
        className={`px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center gap-2 ${
          mode === 'single'
            ? 'bg-emerald-600 text-white'
            : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <Monitor className="w-4 h-4" />
        Single
      </button>
      <button
        onClick={() => onChange('compare')}
        disabled={disabled}
        className={`px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center gap-2 ${
          mode === 'compare'
            ? 'bg-emerald-600 text-white'
            : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <Grid className="w-4 h-4" />
        Compare
      </button>
    </div>
  );
}
