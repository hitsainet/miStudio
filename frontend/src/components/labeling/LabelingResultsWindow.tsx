import React, { useState, useEffect, useRef } from 'react';
import { Copy } from 'lucide-react';
import { LabelingResult, useLabelingResultsWebSocket } from '../../hooks/useLabelingResultsWebSocket';

interface LabelingResultsWindowProps {
  labelingJobId: string | null;
}

export const LabelingResultsWindow: React.FC<LabelingResultsWindowProps> = ({
  labelingJobId
}) => {
  const [results, setResults] = useState<LabelingResult[]>([]);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  console.log('[LabelingResultsWindow] Rendered with labelingJobId:', labelingJobId, 'results count:', results.length);

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
      const tokensText = result.example_tokens.join(', ');
      return `#${result.feature_id} - ${result.label} (${result.category})\nDescription: ${result.description || 'N/A'}\nExample Tokens: ${tokensText}\n`;
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
      <div className="bg-slate-900 border border-slate-700 rounded p-3 h-64">
        <h3 className="text-xs font-medium text-slate-300 mb-2">Recent Labeling Results</h3>
        <div className="flex items-center justify-center h-full text-slate-500 text-xs">
          {labelingJobId ? 'Waiting for results...' : 'Start a labeling job to see live results'}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-900 border border-slate-700 rounded">
      <div className="p-2 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-xs font-medium text-slate-300">Recent Labeling Results</h3>
          <p className="text-xs text-slate-500 mt-0.5">Last {results.length} features</p>
        </div>
        <button
          onClick={handleCopyResults}
          className="p-1.5 rounded hover:bg-slate-800 transition-colors text-slate-400 hover:text-slate-200"
          title="Copy results to clipboard"
        >
          <Copy className="w-4 h-4" />
        </button>
      </div>

      <div ref={scrollContainerRef} className="h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900">
        {results.map((result, index) => (
          <div
            key={`${result.feature_id}-${index}`}
            className="p-2 border-b border-slate-800 hover:bg-slate-850 transition-colors"
          >
            <div className="flex items-start justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-emerald-400">#{result.feature_id}</span>
                <span className="text-xs font-medium text-slate-200">{result.label}</span>
              </div>
              <span className={`text-xs px-1.5 py-0.5 rounded ${getCategoryColor(result.category)}`}>
                {result.category}
              </span>
            </div>

            {result.description && (
              <p className="text-xs text-slate-400 mb-1 line-clamp-1">{result.description}</p>
            )}

            {result.example_tokens.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {result.example_tokens.slice(0, 3).map((token, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-1 py-0.5 bg-slate-800 text-slate-300 rounded font-mono"
                  >
                    {token}
                  </span>
                ))}
                {result.example_tokens.length > 3 && (
                  <span className="text-xs text-slate-500">+{result.example_tokens.length - 3}</span>
                )}
              </div>
            )}
          </div>
        ))}
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
  return colors[category] || 'bg-slate-700 text-slate-300';
}
