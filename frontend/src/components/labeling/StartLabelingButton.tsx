/**
 * StartLabelingButton Component
 *
 * Button to trigger semantic labeling for an extraction job.
 * Opens a modal to configure labeling options.
 */

import React, { useState } from 'react';
import { Tag, AlertCircle } from 'lucide-react';
import { useLabelingStore } from '../../stores/labelingStore';
import { LabelingMethod } from '../../types/labeling';

interface StartLabelingButtonProps {
  extractionId: string;
  disabled?: boolean;
  onSuccess?: () => void;
}

export const StartLabelingButton: React.FC<StartLabelingButtonProps> = ({
  extractionId,
  disabled = false,
  onSuccess,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [labelingMethod, setLabelingMethod] = useState<LabelingMethod>(
    LabelingMethod.OPENAI
  );
  const [openaiModel, setOpenaiModel] = useState('gpt-4o-mini');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [localModel, setLocalModel] = useState('meta-llama/Llama-3.2-1B');

  const { startLabeling, isLoading, error, clearError } = useLabelingStore();

  const handleStartLabeling = async () => {
    try {
      clearError();

      await startLabeling({
        extraction_job_id: extractionId,
        labeling_method: labelingMethod,
        openai_model: labelingMethod === LabelingMethod.OPENAI ? openaiModel : undefined,
        openai_api_key:
          labelingMethod === LabelingMethod.OPENAI && openaiApiKey
            ? openaiApiKey
            : undefined,
        local_model: labelingMethod === LabelingMethod.LOCAL ? localModel : undefined,
        batch_size: 10,
      });

      setIsOpen(false);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to start labeling:', err);
    }
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        disabled={disabled || isLoading}
        className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
      >
        <Tag className="w-4 h-4" />
        Label Features
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-slate-800 rounded-lg p-6 w-full max-w-md border border-slate-700">
            <h2 className="text-xl font-semibold text-white mb-4">
              Start Semantic Labeling
            </h2>

            {error && (
              <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-red-200">{error}</div>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Labeling Method
                </label>
                <select
                  value={labelingMethod}
                  onChange={(e) => setLabelingMethod(e.target.value as LabelingMethod)}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                >
                  <option value={LabelingMethod.OPENAI}>OpenAI (gpt-4o-mini)</option>
                  <option value={LabelingMethod.LOCAL}>Local Model</option>
                </select>
              </div>

              {labelingMethod === LabelingMethod.OPENAI && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      OpenAI Model
                    </label>
                    <select
                      value={openaiModel}
                      onChange={(e) => setOpenaiModel(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value="gpt-4o-mini">gpt-4o-mini (recommended)</option>
                      <option value="gpt-4o">gpt-4o</option>
                      <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      OpenAI API Key (optional)
                    </label>
                    <input
                      type="password"
                      value={openaiApiKey}
                      onChange={(e) => setOpenaiApiKey(e.target.value)}
                      placeholder="Uses server default if not provided"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-400">
                      Leave blank to use the server's configured API key
                    </p>
                  </div>
                </>
              )}

              {labelingMethod === LabelingMethod.LOCAL && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Local Model
                  </label>
                  <input
                    type="text"
                    value={localModel}
                    onChange={(e) => setLocalModel(e.target.value)}
                    placeholder="meta-llama/Llama-3.2-1B"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>
              )}
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={handleStartLabeling}
                disabled={isLoading}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? 'Starting...' : 'Start Labeling'}
              </button>
              <button
                onClick={() => {
                  setIsOpen(false);
                  clearError();
                }}
                disabled={isLoading}
                className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
