/**
 * TemplatesPanel - Unified template management panel with sub-tabs.
 *
 * This panel provides a tabbed interface for managing different types of templates:
 * - Extraction Templates: For activation extraction configurations
 * - Training Templates: For SAE training configurations
 * - Labeling Templates: For semantic feature labeling prompt configurations
 */

import { useState } from 'react';
import { ExtractionTemplatesPanel } from './ExtractionTemplatesPanel';
import { TrainingTemplatesPanel } from './TrainingTemplatesPanel';
import { LabelingPromptTemplatesPanel } from './LabelingPromptTemplatesPanel';
import { PromptTemplatesPanel } from './PromptTemplatesPanel';

type TemplateType = 'extraction' | 'training' | 'labeling' | 'prompt';

export function TemplatesPanel() {
  const [activeTemplateType, setActiveTemplateType] = useState<TemplateType>('extraction');

  return (
    <div className="max-w-[80%] mx-auto px-6 py-8">
      {/* Sub-tabs for template types */}
      <div className="mb-6">
        <div className="border-b border-slate-800">
          <nav className="flex gap-1">
            <button
              onClick={() => setActiveTemplateType('extraction')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activeTemplateType === 'extraction'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Extraction Templates
              {activeTemplateType === 'extraction' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActiveTemplateType('training')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activeTemplateType === 'training'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Training Templates
              {activeTemplateType === 'training' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActiveTemplateType('labeling')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activeTemplateType === 'labeling'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Labeling Templates
              {activeTemplateType === 'labeling' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
            <button
              onClick={() => setActiveTemplateType('prompt')}
              className={`px-6 py-3 font-medium transition-colors relative ${
                activeTemplateType === 'prompt'
                  ? 'text-emerald-400'
                  : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              Steering Prompts
              {activeTemplateType === 'prompt' && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400"></div>
              )}
            </button>
          </nav>
        </div>
      </div>

      {/* Template content based on active sub-tab */}
      <div>
        {activeTemplateType === 'extraction' && <ExtractionTemplatesPanel />}
        {activeTemplateType === 'training' && <TrainingTemplatesPanel />}
        {activeTemplateType === 'labeling' && <LabelingPromptTemplatesPanel />}
        {activeTemplateType === 'prompt' && <PromptTemplatesPanel />}
      </div>
    </div>
  );
}
