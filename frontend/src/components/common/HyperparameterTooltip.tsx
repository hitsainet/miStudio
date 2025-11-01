/**
 * HyperparameterTooltip Component
 *
 * Displays detailed hyperparameter documentation in an accessible popover.
 * Shows purpose, description, examples, recommendations, and warnings.
 */

import React, { useState, useRef, useEffect } from 'react';
import { Info, X, AlertTriangle, Lightbulb, TrendingUp } from 'lucide-react';
import { getHyperparameterDoc } from '../../config/hyperparameterDocs';

interface HyperparameterTooltipProps {
  paramName: string;
  className?: string;
}

export const HyperparameterTooltip: React.FC<HyperparameterTooltipProps> = ({
  paramName,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<'above' | 'below'>('below');
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const doc = getHyperparameterDoc(paramName);

  // Handle click outside to close
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Calculate position (above or below) based on viewport
  useEffect(() => {
    if (!isOpen || !buttonRef.current) return;

    const rect = buttonRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;

    // If less than 400px below, show above
    setPosition(spaceBelow < 400 && spaceAbove > 400 ? 'above' : 'below');
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen]);

  if (!doc) {
    return null; // No documentation available
  }

  return (
    <div className={`relative inline-block ${className}`}>
      {/* Info Icon Button */}
      <button
        ref={buttonRef}
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="inline-flex items-center justify-center w-5 h-5 text-slate-400 hover:text-emerald-400 transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900 rounded-full"
        aria-label={`Show documentation for ${doc.name}`}
      >
        <Info size={18} />
      </button>

      {/* Popover */}
      {isOpen && (
        <div
          ref={popoverRef}
          className={`absolute z-50 w-[480px] max-h-[600px] overflow-y-auto bg-slate-800 border border-slate-700 rounded-lg shadow-2xl ${
            position === 'above' ? 'bottom-full mb-2' : 'top-full mt-2'
          } left-0`}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="sticky top-0 bg-slate-800 border-b border-slate-700 px-4 py-3 flex items-center justify-between">
            <h3 className="text-base font-semibold text-slate-100">{doc.name}</h3>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsOpen(false);
              }}
              className="p-1 hover:bg-slate-700 rounded transition-colors"
              aria-label="Close documentation"
            >
              <X size={16} className="text-slate-400" />
            </button>
          </div>

          {/* Content */}
          <div className="px-4 py-4 space-y-4">
            {/* Purpose */}
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Lightbulb size={16} className="text-emerald-400" />
                <h4 className="text-sm font-semibold text-emerald-400">Purpose</h4>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{doc.purpose}</p>
            </div>

            {/* Description */}
            <div>
              <p className="text-sm text-slate-400 leading-relaxed">{doc.description}</p>
            </div>

            {/* Examples */}
            {doc.examples && doc.examples.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp size={16} className="text-blue-400" />
                  <h4 className="text-sm font-semibold text-blue-400">Examples & Effects</h4>
                </div>
                <div className="space-y-3">
                  {doc.examples.map((example, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-900/50 border border-slate-700 rounded-md p-3"
                    >
                      <div className="flex items-baseline gap-2 mb-1">
                        <code className="text-sm font-mono font-semibold text-emerald-400">
                          {example.value}
                        </code>
                        <span className="text-xs text-slate-500">→</span>
                        <span className="text-sm text-slate-300">{example.effect}</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">{example.useCase}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {doc.recommendations && doc.recommendations.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-slate-300 mb-2">
                  💡 Recommendations
                </h4>
                <ul className="space-y-1.5">
                  {doc.recommendations.map((rec, idx) => (
                    <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                      <span className="text-emerald-400 mt-1">•</span>
                      <span className="flex-1">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Warnings */}
            {doc.warnings && doc.warnings.length > 0 && (
              <div className="bg-orange-900/20 border border-orange-900/50 rounded-md p-3">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle size={16} className="text-orange-400" />
                  <h4 className="text-sm font-semibold text-orange-400">Warnings</h4>
                </div>
                <ul className="space-y-1.5">
                  {doc.warnings.map((warning, idx) => (
                    <li key={idx} className="text-sm text-orange-200 flex items-start gap-2">
                      <span className="text-orange-400 mt-1">⚠</span>
                      <span className="flex-1">{warning}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Related Parameters */}
            {doc.relatedParams && doc.relatedParams.length > 0 && (
              <div className="pt-2 border-t border-slate-700">
                <p className="text-xs text-slate-500">
                  Related:{' '}
                  {doc.relatedParams.map((param, idx) => (
                    <React.Fragment key={param}>
                      {idx > 0 && ', '}
                      <code className="text-slate-400">{param}</code>
                    </React.Fragment>
                  ))}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * HyperparameterLabel Component
 *
 * Convenience component that combines label text with tooltip icon.
 */
interface HyperparameterLabelProps {
  paramName: string;
  label: string;
  required?: boolean;
  htmlFor?: string;
  className?: string;
}

export const HyperparameterLabel: React.FC<HyperparameterLabelProps> = ({
  paramName,
  label,
  required = false,
  htmlFor,
  className = '',
}) => {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <label htmlFor={htmlFor} className="block text-sm font-medium text-slate-300">
        {label}
        {required && <span className="text-red-400 ml-1">*</span>}
      </label>
      <HyperparameterTooltip paramName={paramName} />
    </div>
  );
};
