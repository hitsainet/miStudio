/**
 * PromptListEditor - Multi-prompt input editor for batch steering.
 *
 * Features:
 * - Add/remove prompts (minimum 1)
 * - Edit each prompt individually
 * - Visual indication of empty prompts
 * - Clear all prompts action
 * - Compact header with prompt count
 * - Multi-line paste detection: automatically splits pasted text into multiple prompts
 * - Save prompts as templates
 * - Load prompts from templates
 */

import { useState } from 'react';
import { Plus, X, Trash2, Save, FolderOpen } from 'lucide-react';
import { PromptTemplate } from '../../types/promptTemplate';

interface PromptListEditorProps {
  prompts: string[];
  onAddPrompt: () => void;
  onRemovePrompt: (index: number) => void;
  onUpdatePrompt: (index: number, value: string) => void;
  onClearPrompts: () => void;
  onReplacePromptWithMultiple: (index: number, newPrompts: string[]) => void;
  // Template operations
  onSaveAsTemplate?: (name: string, description: string) => Promise<void>;
  onLoadTemplate?: (prompts: string[]) => void;
  availableTemplates?: PromptTemplate[];
  onFetchTemplates?: () => void;
  disabled?: boolean;
}

export function PromptListEditor({
  prompts,
  onAddPrompt,
  onRemovePrompt,
  onUpdatePrompt,
  onClearPrompts,
  onReplacePromptWithMultiple,
  onSaveAsTemplate,
  onLoadTemplate,
  availableTemplates = [],
  onFetchTemplates,
  disabled = false,
}: PromptListEditorProps) {
  const nonEmptyCount = prompts.filter((p) => p.trim().length > 0).length;
  const canRemove = prompts.length > 1;

  // Modal states
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [showLoadModal, setShowLoadModal] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDescription, setTemplateDescription] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Template operations enabled
  const canSave = onSaveAsTemplate && nonEmptyCount > 0;

  const handleOpenSaveModal = () => {
    setTemplateName('');
    setTemplateDescription('');
    setSaveError(null);
    setShowSaveModal(true);
  };

  const handleSaveTemplate = async () => {
    if (!templateName.trim() || !onSaveAsTemplate) return;

    setIsSaving(true);
    setSaveError(null);
    try {
      await onSaveAsTemplate(templateName.trim(), templateDescription.trim());
      setShowSaveModal(false);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save template');
    } finally {
      setIsSaving(false);
    }
  };

  const handleOpenLoadModal = () => {
    onFetchTemplates?.();
    setShowLoadModal(true);
  };

  const handleLoadTemplate = (template: PromptTemplate) => {
    onLoadTemplate?.(template.prompts);
    setShowLoadModal(false);
  };

  /**
   * Parse multi-line content from prompts.
   * Returns the index of the first prompt with multi-line content and the parsed lines,
   * or null if no multi-line content is found.
   */
  const findMultiLinePrompt = (): { index: number; lines: string[] } | null => {
    for (let i = 0; i < prompts.length; i++) {
      const prompt = prompts[i];
      if (prompt.includes('\n')) {
        // Split by newlines, trim each line, filter out empty/whitespace-only lines
        const lines = prompt
          .split('\n')
          .map((line) => line.trim())
          .filter((line) => line.length > 0);

        if (lines.length > 1) {
          return { index: i, lines };
        }
      }
    }
    return null;
  };

  /**
   * Handle Add Prompt button click.
   * If any prompt contains multi-line content, parse and split it into multiple prompts.
   * Otherwise, add a single empty prompt.
   */
  const handleAddPromptClick = () => {
    const multiLineResult = findMultiLinePrompt();

    if (multiLineResult) {
      // Found multi-line content - replace that prompt with multiple prompts
      onReplacePromptWithMultiple(multiLineResult.index, multiLineResult.lines);
    } else {
      // No multi-line content - add empty prompt (default behavior)
      onAddPrompt();
    }
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-slate-300">
          Prompts
          {prompts.length > 1 && (
            <span className="ml-2 text-slate-500 font-normal">
              ({nonEmptyCount} of {prompts.length} with content)
            </span>
          )}
        </label>
        <div className="flex items-center gap-2">
          {/* Load Template button */}
          {onLoadTemplate && (
            <button
              onClick={handleOpenLoadModal}
              disabled={disabled}
              className="text-xs text-slate-400 hover:text-slate-300 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Load prompts from template"
            >
              <FolderOpen className="w-3 h-3" />
              Load
            </button>
          )}
          {/* Save as Template button */}
          {canSave && (
            <button
              onClick={handleOpenSaveModal}
              disabled={disabled}
              className="text-xs text-slate-400 hover:text-slate-300 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Save prompts as template"
            >
              <Save className="w-3 h-3" />
              Save
            </button>
          )}
          {prompts.length > 1 && (
            <button
              onClick={onClearPrompts}
              disabled={disabled}
              className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Clear all prompts"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          )}
          <button
            onClick={handleAddPromptClick}
            disabled={disabled}
            className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Add prompt (or split multi-line content into multiple prompts)"
          >
            <Plus className="w-3 h-3" />
            Add Prompt
          </button>
        </div>
      </div>

      {/* Prompt inputs */}
      <div className="space-y-2">
        {prompts.map((prompt, index) => (
          <div key={index} className="relative group">
            <div className="flex items-start gap-2">
              {/* Prompt number indicator */}
              {prompts.length > 1 && (
                <div className="flex-shrink-0 w-6 h-6 mt-3 flex items-center justify-center rounded bg-slate-800 text-slate-500 text-xs font-mono">
                  {index + 1}
                </div>
              )}

              {/* Textarea */}
              <textarea
                value={prompt}
                onChange={(e) => onUpdatePrompt(index, e.target.value)}
                placeholder={
                  prompts.length === 1
                    ? 'Enter your prompt here... (paste multiple lines and click Add Prompt to split)'
                    : `Prompt ${index + 1}...`
                }
                rows={prompts.length === 1 ? 4 : 2}
                disabled={disabled}
                className={`flex-1 px-4 py-3 bg-slate-900 border rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 resize-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  prompt.trim().length === 0 && prompts.length > 1
                    ? 'border-amber-500/30'
                    : 'border-slate-700'
                }`}
              />

              {/* Remove button */}
              {canRemove && (
                <button
                  onClick={() => onRemovePrompt(index)}
                  disabled={disabled}
                  className="flex-shrink-0 p-2 mt-2 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Remove this prompt"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Multi-line hint */}
            {prompt.includes('\n') && prompt.split('\n').filter((l) => l.trim()).length > 1 && (
              <p className="text-xs text-emerald-500/70 mt-1 ml-8">
                Contains multiple lines - click "Add Prompt" to split into separate prompts
              </p>
            )}

            {/* Empty prompt warning */}
            {prompt.trim().length === 0 && prompts.length > 1 && (
              <p className="text-xs text-amber-500/70 mt-1 ml-8">
                Empty prompts will be skipped during batch processing
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Batch mode hint */}
      {prompts.length > 1 && nonEmptyCount > 1 && (
        <p className="text-xs text-slate-500">
          Batch mode: Each prompt will be processed sequentially with the same feature configuration.
        </p>
      )}

      {/* Save Template Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setShowSaveModal(false)}
          />
          <div className="relative bg-slate-900 border border-slate-700 rounded-lg p-6 w-full max-w-md shadow-xl">
            <h3 className="text-lg font-medium text-slate-100 mb-4">Save as Steering Prompt Template</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Template Name *
                </label>
                <input
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="Enter template name..."
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">
                  Description (optional)
                </label>
                <textarea
                  value={templateDescription}
                  onChange={(e) => setTemplateDescription(e.target.value)}
                  placeholder="Brief description of this prompt set..."
                  rows={2}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 resize-none"
                />
              </div>

              <p className="text-sm text-slate-500">
                {nonEmptyCount} prompt{nonEmptyCount !== 1 ? 's' : ''} will be saved
              </p>

              {saveError && (
                <p className="text-sm text-red-400">{saveError}</p>
              )}
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowSaveModal(false)}
                className="px-4 py-2 text-slate-300 hover:text-slate-100 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveTemplate}
                disabled={!templateName.trim() || isSaving}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSaving ? 'Saving...' : 'Save Template'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Load Template Modal */}
      {showLoadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setShowLoadModal(false)}
          />
          <div className="relative bg-slate-900 border border-slate-700 rounded-lg p-6 w-full max-w-lg shadow-xl max-h-[70vh] overflow-hidden flex flex-col">
            <h3 className="text-lg font-medium text-slate-100 mb-4">Load Steering Prompt Template</h3>

            {availableTemplates.length === 0 ? (
              <p className="text-slate-400 text-center py-8">
                No steering prompt templates available. Create templates from the Templates page.
              </p>
            ) : (
              <div className="overflow-y-auto flex-1 space-y-2">
                {availableTemplates.map((template) => (
                  <button
                    key={template.id}
                    onClick={() => handleLoadTemplate(template)}
                    className="w-full text-left p-4 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
                  >
                    <div className="font-medium text-slate-100">{template.name}</div>
                    {template.description && (
                      <p className="text-sm text-slate-400 mt-1 line-clamp-2">
                        {template.description}
                      </p>
                    )}
                    <p className="text-xs text-slate-500 mt-2">
                      {template.prompts.length} prompt{template.prompts.length !== 1 ? 's' : ''}
                    </p>
                  </button>
                ))}
              </div>
            )}

            <div className="flex justify-end mt-4 pt-4 border-t border-slate-700">
              <button
                onClick={() => setShowLoadModal(false)}
                className="px-4 py-2 text-slate-300 hover:text-slate-100 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
