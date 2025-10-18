/**
 * SettingsModal Component
 *
 * Modal for configuring System Monitor settings
 */

import { X, Clock } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  updateInterval: number;
  onUpdateIntervalChange: (interval: number) => void;
}

export function SettingsModal({
  isOpen,
  onClose,
  updateInterval,
  onUpdateIntervalChange,
}: SettingsModalProps) {
  if (!isOpen) return null;

  const intervals = [
    { value: 500, label: '0.5s (High CPU)' },
    { value: 1000, label: '1s (Recommended)' },
    { value: 2000, label: '2s' },
    { value: 5000, label: '5s' },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-md w-full">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <h2 className="text-xl font-semibold text-slate-100">Settings</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Update Interval */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Clock className="w-5 h-5 text-emerald-400" />
              <label className="text-sm font-medium text-slate-100">
                Update Interval
              </label>
            </div>
            <p className="text-xs text-slate-400 mb-3">
              How often metrics are refreshed. Lower values use more CPU.
            </p>
            <div className="space-y-2">
              {intervals.map((interval) => (
                <label
                  key={interval.value}
                  className="flex items-center gap-3 p-3 rounded-lg border border-slate-700 hover:border-slate-600 cursor-pointer transition-colors"
                >
                  <input
                    type="radio"
                    name="interval"
                    value={interval.value}
                    checked={updateInterval === interval.value}
                    onChange={() => onUpdateIntervalChange(interval.value)}
                    className="w-4 h-4 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                  />
                  <span className="text-sm text-slate-200">{interval.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Info */}
          <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
            <p className="text-xs text-slate-400">
              Changes take effect immediately. Settings are saved locally in your browser.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors text-white font-medium"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
