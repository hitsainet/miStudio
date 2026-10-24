/**
 * ErrorBanner Component
 *
 * Displays error messages and connection status at the top of the System Monitor
 */

import { AlertCircle, WifiOff, RefreshCw } from 'lucide-react';

export type ErrorType = 'connection' | 'gpu' | 'api' | 'general';

interface ErrorBannerProps {
  type: ErrorType;
  message: string;
  isRetrying?: boolean;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export function ErrorBanner({
  type,
  message,
  isRetrying = false,
  onRetry,
  onDismiss
}: ErrorBannerProps) {
  const getIcon = () => {
    switch (type) {
      case 'connection':
        return <WifiOff className="w-5 h-5" />;
      case 'gpu':
      case 'api':
      case 'general':
      default:
        return <AlertCircle className="w-5 h-5" />;
    }
  };

  const getStyles = () => {
    switch (type) {
      case 'connection':
        return 'bg-yellow-900/50 border-yellow-700 text-yellow-200';
      case 'gpu':
        return 'bg-orange-900/50 border-orange-700 text-orange-200';
      case 'api':
      case 'general':
      default:
        return 'bg-red-900/50 border-red-700 text-red-200';
    }
  };

  return (
    <div className={`rounded-lg border p-4 ${getStyles()} animate-in slide-in-from-top duration-300`}>
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {getIcon()}
          <div>
            <p className="font-medium">{message}</p>
            {type === 'connection' && (
              <p className="text-sm opacity-80 mt-1">
                Displaying last known values. Attempting to reconnect...
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onRetry && (
            <button
              onClick={onRetry}
              disabled={isRetrying}
              className="px-3 py-1.5 rounded-md bg-white/10 hover:bg-white/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
            >
              <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
              {isRetrying ? 'Retrying...' : 'Retry'}
            </button>
          )}

          {onDismiss && (
            <button
              onClick={onDismiss}
              className="px-3 py-1.5 rounded-md bg-white/10 hover:bg-white/20 transition-colors text-sm"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
