import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Copy, GripHorizontal, Maximize2, Minimize2 } from 'lucide-react';
import { LabelingResult, useLabelingResultsWebSocket } from '../../hooks/useLabelingResultsWebSocket';
import { joinTokensWithProperSpacing } from '../../utils/tokenDisplay';

const MIN_HEIGHT = 128;  // 8rem
const DEFAULT_HEIGHT = 256; // 16rem (h-64)
const MAX_HEIGHT = 800;

interface LabelingResultsWindowProps {
  labelingJobId: string | null;
}

export const LabelingResultsWindow: React.FC<LabelingResultsWindowProps> = ({
  labelingJobId
}) => {
  const [results, setResults] = useState<LabelingResult[]>([]);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [isMaximized, setIsMaximized] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isResizing = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const previousHeight = useRef(DEFAULT_HEIGHT);

  console.log('[LabelingResultsWindow] Rendered with labelingJobId:', labelingJobId, 'results count:', results.length);

  // Toggle maximize/restore
  const toggleMaximize = useCallback(() => {
    if (isMaximized) {
      setHeight(previousHeight.current);
      setIsMaximized(false);
    } else {
      previousHeight.current = height;
      setHeight(MAX_HEIGHT);
      setIsMaximized(true);
    }
  }, [isMaximized, height]);

  // Drag-to-resize handlers
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    startY.current = e.clientY;
    startHeight.current = height;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';

    const handleResizeMove = (moveEvent: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = moveEvent.clientY - startY.current;
      const newHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, startHeight.current + delta));
      setHeight(newHeight);
      setIsMaximized(false);
    };

    const handleResizeEnd = () => {
      isResizing.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
    };

    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
  }, [height]);

  // Subscribe to real-time results
  useLabelingResultsWebSocket(labelingJobId, (result) => {
    console.log('[LabelingResultsWindow] Callback invoked with result:', result);
    setResults((prev) => {
      const newResults = [result, ...prev];
      // Keep only last 20 results
      return newResults.slice(0, 20);
    });
  });

  // Auto-scroll to top when new result arrives (only scroll within the container)
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = 0;
    }
  }, [results]);

  // Clear results when job changes
  useEffect(() => {
    setResults([]);
  }, [labelingJobId]);

  // Copy results to clipboard
  const handleCopyResults = async () => {
    if (results.length === 0) {
      console.log('[LabelingResultsWindow] No results to copy');
      return;
    }

    const resultsText = results.map((result) => {
      const examples = result.examples || [];
      let text = `#${result.feature_id} - ${result.label} (${result.category})\n`;
      text += `Description: ${result.description || 'N/A'}\n`;

      if (examples.length > 0) {
        text += 'Examples:\n';
        examples.slice(0, 10).forEach((ex, idx) => {
          const prefix = joinTokensWithProperSpacing(ex.prefix_tokens?.slice(-10) || []);
          const prime = ex.prime_token || '';
          const suffix = joinTokensWithProperSpacing(ex.suffix_tokens?.slice(0, 10) || []);
          // Add space before prime if it starts a new word
          const spaceBeforePrime = prime.startsWith('Ġ') || prime.startsWith(' ') ? ' ' : '';
          // Add space before suffix if it starts a new word
          const spaceBeforeSuffix = (ex.suffix_tokens?.[0]?.startsWith('Ġ') || ex.suffix_tokens?.[0]?.startsWith(' ')) ? ' ' : '';
          text += `${String(idx + 1).padStart(2, '0')}: ${prefix}${spaceBeforePrime}${prime}${spaceBeforeSuffix}${suffix}\n`;
        });
        if (examples.length > 10) {
          text += `... +${examples.length - 10} more examples\n`;
        }
      }

      return text;
    }).join('\n');

    console.log('[LabelingResultsWindow] Attempting to copy:', resultsText.substring(0, 200));

    try {
      await navigator.clipboard.writeText(resultsText);
      console.log('[LabelingResultsWindow] Successfully copied to clipboard');
    } catch (err) {
      console.error('[LabelingResultsWindow] Failed to copy results:', err);
      // Fallback method for older browsers or permission issues
      try {
        const textArea = document.createElement('textarea');
        textArea.value = resultsText;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        console.log('[LabelingResultsWindow] Successfully copied using fallback method');
      } catch (fallbackErr) {
        console.error('[LabelingResultsWindow] Fallback copy also failed:', fallbackErr);
      }
    }
  };

  if (!labelingJobId || results.length === 0) {
    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded p-3" style={{ height }}>
        <h3 className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-2">Recent Labeling Results</h3>
        <div className="flex items-center justify-center h-full text-slate-500 text-xs">
          {labelingJobId ? 'Waiting for results...' : 'Start a labeling job to see live results'}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded">
      <div className="p-2 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-xs font-medium text-slate-700 dark:text-slate-300">Recent Labeling Results</h3>
          <p className="text-xs text-slate-500 mt-0.5">Last {results.length} features</p>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleMaximize}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            title={isMaximized ? 'Restore default height' : 'Maximize height'}
          >
            {isMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button
            onClick={handleCopyResults}
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            title="Copy results to clipboard"
          >
            <Copy className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div ref={scrollContainerRef} className="overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900" style={{ height }}>
        {results.map((result, index) => {
          const examples = result.examples || [];

          return (
            <div
              key={`${result.feature_id}-${index}`}
              className="p-2 border-b border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-850 transition-colors"
            >
              <div className="flex items-start justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-emerald-400">#{result.feature_id}</span>
                  <span className="text-xs font-medium text-slate-800 dark:text-slate-200">{result.label}</span>
                </div>
                <span className={`text-xs px-1.5 py-0.5 rounded ${getCategoryColor(result.category)}`}>
                  {result.category}
                </span>
              </div>

              {result.description && (
                <p className="text-xs text-slate-600 dark:text-slate-400 mb-1 line-clamp-2">{result.description}</p>
              )}

              {/* Display numbered list of examples with prefix, prime, and suffix tokens */}
              {examples.length > 0 && (
                <div className="space-y-0.5 mt-2">
                  {examples.slice(0, 10).map((example, idx) => {
                    const prefix = joinTokensWithProperSpacing(example.prefix_tokens?.slice(-10) || []);
                    const prime = example.prime_token || '';
                    const suffix = joinTokensWithProperSpacing(example.suffix_tokens?.slice(0, 10) || []);
                    // Check if we need spaces around the prime token
                    const spaceBeforePrime = prime.startsWith('Ġ') || prime.startsWith(' ');
                    const spaceBeforeSuffix = example.suffix_tokens?.[0]?.startsWith('Ġ') || example.suffix_tokens?.[0]?.startsWith(' ');
                    // Clean the prime token of space markers for display
                    const cleanPrime = prime.replace(/^Ġ/, '').trim() || prime;

                    return (
                      <div key={idx} className="text-xs font-mono text-slate-700 dark:text-slate-300">
                        <span className="text-slate-500">{String(idx + 1).padStart(2, '0')}:</span>{' '}
                        <span className="text-slate-600 dark:text-slate-400">{prefix}</span>
                        {spaceBeforePrime && prefix && <span> </span>}
                        <span className="text-emerald-400 font-semibold">{cleanPrime}</span>
                        {spaceBeforeSuffix && suffix && <span> </span>}
                        <span className="text-slate-600 dark:text-slate-400">{suffix}</span>
                      </div>
                    );
                  })}
                  {examples.length > 10 && (
                    <div className="text-xs text-slate-500 italic">
                      +{examples.length - 10} more examples
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Resize handle */}
      <div
        onMouseDown={handleResizeStart}
        className="flex items-center justify-center h-3 cursor-row-resize border-t border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-colors group"
      >
        <GripHorizontal className="w-4 h-3 text-slate-600 group-hover:text-slate-400" />
      </div>
    </div>
  );
};

function getCategoryColor(category: string): string {
  const colors: Record<string, string> = {
    semantic: 'bg-blue-900/30 text-blue-300',
    syntactic: 'bg-purple-900/30 text-purple-300',
    structural: 'bg-yellow-900/30 text-yellow-300',
    positional: 'bg-green-900/30 text-green-300',
    morphological: 'bg-pink-900/30 text-pink-300',
    mixed: 'bg-orange-900/30 text-orange-300',
  };
  return colors[category] || 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300';
}
