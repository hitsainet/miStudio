/**
 * TemplateSelector Component
 *
 * Dropdown selector for loading training templates that match the currently
 * selected model and dataset. Positioned next to the extraction dropdown.
 */

import React, { useEffect, useState } from 'react';
import { BookTemplate, X } from 'lucide-react';
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

  // Track selected template ID for controlled component
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('');

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

  // Get selected template name for display
  const selectedTemplate = selectedTemplateId
    ? matchingTemplates.find((t) => t.id === selectedTemplateId)
    : null;

  const handleTemplateSelect = (templateId: string) => {
    const template = templates.find((t) => t.id === templateId);
    if (template) {
      setSelectedTemplateId(templateId);
      onTemplateLoad(template);
    }
  };

  const handleClearSelection = () => {
    setSelectedTemplateId('');
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
          <div className="relative">
            <select
              id="template-selector"
              value={selectedTemplateId}
              onChange={(e) => {
                if (e.target.value) {
                  handleTemplateSelect(e.target.value);
                }
              }}
              className={`w-full px-3 py-2 bg-slate-800 border rounded-md text-sm focus:outline-none focus:border-emerald-500 transition-colors ${
                selectedTemplateId
                  ? 'border-emerald-500/50 text-emerald-300 pr-8'
                  : 'border-slate-700 text-slate-100'
              }`}
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
            {selectedTemplateId && (
              <button
                type="button"
                onClick={handleClearSelection}
                className="absolute right-8 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-slate-200 transition-colors"
                title="Clear selection"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {selectedTemplate
              ? `Loaded: ${selectedTemplate.name}`
              : 'Pre-fill training configuration from a saved template'}
          </p>
        </div>
      </div>
    </div>
  );
};
