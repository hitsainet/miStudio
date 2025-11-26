/**
 * TemplateSelector Component
 *
 * Dropdown selector for loading training templates that match the currently
 * selected model and dataset. Positioned next to the extraction dropdown.
 */

import React, { useEffect } from 'react';
import { BookTemplate } from 'lucide-react';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';
import type { TrainingTemplate } from '../../types/trainingTemplate';

interface TemplateSelectorProps {
  modelId: string;
  datasetId: string;
  onTemplateLoad: (template: TrainingTemplate) => void;
  className?: string;
}

export const TemplateSelector: React.FC<TemplateSelectorProps> = ({
  modelId,
  datasetId,
  onTemplateLoad,
  className = '',
}) => {
  const { templates, fetchTemplates, loading } = useTrainingTemplatesStore();

  // Fetch templates on mount
  useEffect(() => {
    fetchTemplates();
  }, [fetchTemplates]);

  // Filter templates that match the selected model and dataset
  const matchingTemplates = templates.filter((template) => {
    // Match if template has the same model_id and dataset_id
    // Or if template has no model_id/dataset_id (generic templates)
    const modelMatch = !template.model_id || template.model_id === modelId;
    const datasetMatch = !template.dataset_id || template.dataset_id === datasetId;
    return modelMatch && datasetMatch;
  });

  const handleTemplateSelect = (templateId: string) => {
    const template = templates.find((t) => t.id === templateId);
    if (template) {
      onTemplateLoad(template);
    }
  };

  // Don't show if no templates available or still loading
  if (loading || matchingTemplates.length === 0) {
    return null;
  }

  return (
    <div className={className}>
      <div className="flex items-center gap-3">
        <BookTemplate className="w-5 h-5 text-slate-400 flex-shrink-0" />
        <div className="flex-1">
          <label htmlFor="template-selector" className="block text-sm font-medium text-slate-300 mb-1">
            Load Template
          </label>
          <select
            id="template-selector"
            onChange={(e) => {
              if (e.target.value) {
                handleTemplateSelect(e.target.value);
                // Reset select to placeholder after loading
                e.target.value = '';
              }
            }}
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
            defaultValue=""
          >
            <option value="" disabled>
              Select a template... ({matchingTemplates.length} available)
            </option>
            {matchingTemplates.map((template) => (
              <option key={template.id} value={template.id}>
                {template.name}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-slate-500">
            Pre-fill training configuration from a saved template
          </p>
        </div>
      </div>
    </div>
  );
};
