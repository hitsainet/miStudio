/**
 * Extraction Failure Alert Component
 *
 * Displays an inline alert banner when activation extraction fails.
 * Shows error message, suggested retry parameters, and action buttons.
 * Designed to appear within model tiles for immediate visibility.
 */

import { AlertCircle, RefreshCw, X } from 'lucide-react';
import { Model } from '../types/model';

interface ExtractionFailureAlertProps {
  model: Model;
  onRetry: (modelId: string, extractionId: string, retryParams?: {batch_size?: number; max_samples?: number}) => void;
  onCancel: (modelId: string, extractionId: string) => void;
  onDismiss?: () => void;
}

export function ExtractionFailureAlert({ model, onRetry, onCancel, onDismiss }: ExtractionFailureAlertProps) {
  const {
    extraction_id,
    extraction_message,
    extraction_error_type,
    extraction_suggested_retry_params,
  } = model;

  if (!extraction_id || model.extraction_status !== 'failed') {
    return null;
  }

  // Format error message for display
  const getErrorTitle = (errorType?: string) => {
    switch (errorType) {
      case 'OOM':
        return 'Out of Memory';
      case 'VALIDATION':
        return 'Validation Error';
      case 'TIMEOUT':
        return 'Timeout Error';
      case 'EXTRACTION':
        return 'Extraction Error';
      default:
        return 'Extraction Failed';
    }
  };

  // Format suggested parameters for display
  const getSuggestedParams = () => {
    if (!extraction_suggested_retry_params) return null;

    const params = [];
    if (extraction_suggested_retry_params.batch_size) {
      params.push(`batch_size: ${extraction_suggested_retry_params.batch_size}`);
    }
    if (extraction_suggested_retry_params.max_samples) {
      params.push(`max_samples: ${extraction_suggested_retry_params.max_samples}`);
    }
    return params.length > 0 ? params.join(', ') : null;
  };

  const suggestedParams = getSuggestedParams();

  return (
    <div className="bg-red-900/20 border border-red-700 rounded-md p-3 mt-2">
      <div className="flex items-start gap-2">
        <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <h4 className="text-sm font-medium text-red-300">
              {getErrorTitle(extraction_error_type)}
            </h4>
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="text-red-400 hover:text-red-300 transition-colors"
                aria-label="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <p className="text-xs text-red-200/80 mb-2 break-words">
            {extraction_message || 'Extraction failed'}
          </p>

          {suggestedParams && (
            <p className="text-xs text-emerald-300/80 mb-2 font-mono">
              Suggested: {suggestedParams}
            </p>
          )}

          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={() => onRetry(model.id, extraction_id, extraction_suggested_retry_params)}
              className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium rounded transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry{suggestedParams ? ' (with suggested params)' : ''}
            </button>

            <button
              onClick={() => onCancel(model.id, extraction_id)}
              className="px-2.5 py-1 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-medium rounded transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
