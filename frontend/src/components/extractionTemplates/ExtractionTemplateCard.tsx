/**
 * ExtractionTemplateCard component for displaying individual template information.
 *
 * This component renders a card with template details and actions.
 */

import React from 'react';
import { Layers, Star, Edit2, Trash2, Copy } from 'lucide-react';
import { ExtractionTemplate } from '../../types/extractionTemplate';

interface ExtractionTemplateCardProps {
  template: ExtractionTemplate;
  onClick?: () => void;
  onEdit?: (template: ExtractionTemplate) => void;
  onDelete?: (id: string) => void;
  onToggleFavorite?: (id: string) => void;
  onDuplicate?: (template: ExtractionTemplate) => void;
}

export function ExtractionTemplateCard({
  template,
  onClick,
  onEdit,
  onDelete,
  onToggleFavorite,
  onDuplicate,
}: ExtractionTemplateCardProps) {
  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    onEdit?.(template);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to delete "${template.name}"?`)) {
      onDelete?.(template.id);
    }
  };

  const handleToggleFavorite = (e: React.MouseEvent) => {
    e.stopPropagation();
    onToggleFavorite?.(template.id);
  };

  const handleDuplicate = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDuplicate?.(template);
  };

  return (
    <div
      className={`bg-slate-900/50 border border-slate-800 rounded-lg p-6 transition-all ${
        onClick ? 'cursor-pointer hover:bg-slate-900/70 hover:border-slate-700' : 'cursor-default'
      }`}
      onClick={onClick}
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <Layers className="w-8 h-8 text-slate-400" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-semibold text-slate-100 truncate">
                  {template.name}
                </h3>
                {template.is_favorite && (
                  <Star className="w-4 h-4 text-yellow-400 fill-yellow-400 flex-shrink-0" />
                )}
              </div>
              {template.description && (
                <p className="text-sm text-slate-400 mt-1 line-clamp-2">{template.description}</p>
              )}
            </div>

            <div className="flex items-center gap-1 flex-shrink-0">
              <button
                onClick={handleToggleFavorite}
                className={`p-1.5 rounded transition-colors ${
                  template.is_favorite
                    ? 'text-yellow-400 hover:bg-yellow-500/10'
                    : 'text-slate-500 hover:bg-slate-700'
                }`}
                title={template.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
              >
                <Star className={`w-4 h-4 ${template.is_favorite ? 'fill-yellow-400' : ''}`} />
              </button>

              {onDuplicate && (
                <button
                  onClick={handleDuplicate}
                  className="p-1.5 hover:bg-slate-700 rounded transition-colors"
                  title="Duplicate template"
                >
                  <Copy className="w-4 h-4 text-slate-400" />
                </button>
              )}

              {onEdit && (
                <button
                  onClick={handleEdit}
                  className="p-1.5 hover:bg-slate-700 rounded transition-colors"
                  title="Edit template"
                >
                  <Edit2 className="w-4 h-4 text-slate-400" />
                </button>
              )}

              {onDelete && (
                <button
                  onClick={handleDelete}
                  className="p-1.5 hover:bg-red-500/10 rounded transition-colors group"
                  title="Delete template"
                >
                  <Trash2 className="w-4 h-4 text-slate-500 group-hover:text-red-400 transition-colors" />
                </button>
              )}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-4 text-sm text-slate-400">
            <div>
              <span className="font-medium">Layer(s):</span> {template.layer_indices.length} layer
              {template.layer_indices.length !== 1 ? 's' : ''}
              <span className="text-slate-500 ml-1">
                ({template.layer_indices.slice(0, 3).join(', ')}
                {template.layer_indices.length > 3 && '...'})
              </span>
            </div>

            <div>
              <span className="font-medium">Hooks:</span> {template.hook_types.join(', ')}
            </div>

            <div>
              <span className="font-medium">Max Samples:</span> {template.max_samples.toLocaleString()}
            </div>

            <div>
              <span className="font-medium">Batch Size:</span> {template.batch_size}
            </div>

            <div>
              <span className="font-medium">Top-K:</span> {template.top_k_examples}
            </div>
          </div>

          {template.extra_metadata && Object.keys(template.extra_metadata).length > 0 && (
            <div className="mt-3 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded text-xs text-slate-400">
              <span className="font-medium">Metadata:</span>{' '}
              {Object.keys(template.extra_metadata).length} field
              {Object.keys(template.extra_metadata).length !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
