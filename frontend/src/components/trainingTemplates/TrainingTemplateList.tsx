/**
 * TrainingTemplateList component for displaying a list of training templates.
 *
 * Renders a grid of template cards with search, filters (architecture type and
 * creation date range) and pagination. Filters are CONTROLLED by the panel and
 * sent to GET /training-templates: the list is paginated server-side, so a
 * browser-side filter would filter only the page in hand.
 */

import { useEffect, useRef, useState } from 'react';
import { Search, SlidersHorizontal, Loader, ChevronLeft, ChevronRight } from 'lucide-react';
import { TrainingTemplate } from '../../types/trainingTemplate';
import { SAEArchitectureType } from '../../types/training';
import { TrainingTemplateCard } from './TrainingTemplateCard';
import { TrainingTemplateFilters, trainingFiltersActive } from '../../utils/templateListFilters';

/** Architecture filter options; 'standard' is the legacy alias of standard_saelens. */
export const ARCHITECTURE_FILTER_OPTIONS: { value: string; label: string }[] = [
  { value: SAEArchitectureType.STANDARD_SAELENS, label: 'Standard (SAELens)' },
  { value: SAEArchitectureType.STANDARD_ANTHROPIC, label: 'Standard (Anthropic)' },
  { value: SAEArchitectureType.JUMPRELU, label: 'JumpReLU' },
  { value: SAEArchitectureType.TOPK, label: 'TopK' },
  { value: SAEArchitectureType.SKIP, label: 'Skip' },
  { value: SAEArchitectureType.TRANSCODER, label: 'Transcoder' },
];

/** Debounce for the search box, so typing sends one request, not one per key. */
export const SEARCH_DEBOUNCE_MS = 300;

const FIELD = 'px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-control text-sm text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-2 focus:ring-emerald-500';

interface TrainingTemplateListProps {
  templates: TrainingTemplate[];
  loading?: boolean;
  onTemplateClick?: (template: TrainingTemplate) => void;
  onEdit?: (template: TrainingTemplate) => void;
  onDelete?: (id: string) => void;
  onToggleFavorite?: (id: string) => void;
  onDuplicate?: (template: TrainingTemplate) => void;
  /** Active filters (controlled) and their setter. */
  filters?: TrainingTemplateFilters;
  onFiltersChange?: (filters: TrainingTemplateFilters) => void;
  /** Server-side total across all pages, for the results count. */
  totalCount?: number;
  // Pagination
  currentPage?: number;
  totalPages?: number;
  hasNext?: boolean;
  hasPrev?: boolean;
  onPageChange?: (page: number) => void;
}

export function TrainingTemplateList({
  templates,
  loading = false,
  onTemplateClick,
  onEdit,
  onDelete,
  onToggleFavorite,
  onDuplicate,
  filters = {},
  onFiltersChange,
  totalCount,
  currentPage = 1,
  totalPages = 1,
  hasNext = false,
  hasPrev = false,
  onPageChange,
}: TrainingTemplateListProps) {
  const [searchText, setSearchText] = useState(filters.search ?? '');
  const [showFilters, setShowFilters] = useState(
    !!(filters.encoder_type || filters.created_from || filters.created_to));
  const active = trainingFiltersActive(filters);

  // Latest filters/setter, so the debounced search never writes a stale copy.
  const latest = useRef({ filters, onFiltersChange });
  latest.current = { filters, onFiltersChange };

  useEffect(() => {
    if ((latest.current.filters.search ?? '') === searchText) return;
    const t = setTimeout(() => {
      latest.current.onFiltersChange?.({ ...latest.current.filters, search: searchText });
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [searchText]);

  const set = (patch: Partial<TrainingTemplateFilters>) =>
    onFiltersChange?.({ ...filters, search: searchText, ...patch });

  const clearAll = () => {
    setSearchText('');
    onFiltersChange?.({});
  };

  if (!loading && templates.length === 0 && !active) {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white dark:bg-slate-800 mb-4">
          <Search className="w-8 h-8 text-slate-500" />
        </div>
        <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">No templates found</h3>
        <p className="text-sm text-slate-500">
          Create your first training template to get started.
        </p>
      </div>
    );
  }

  const filterCount = [filters.encoder_type, filters.created_from, filters.created_to]
    .filter(Boolean).length;
  const total = totalCount ?? templates.length;

  return (
    <div>
      {/* Search and Filters — always mounted, so typing keeps focus across reloads */}
      <div className="mb-6 flex gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            aria-label="Search templates"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search templates by name or description..."
            className="w-full pl-10 pr-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-control text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          aria-expanded={showFilters}
          className={`flex items-center gap-2 px-4 py-2 rounded-control transition-colors ${
            showFilters
              ? 'bg-emerald-600 text-white'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
          }`}
        >
          <SlidersHorizontal className="w-4 h-4" />
          Filters{filterCount > 0 ? ` (${filterCount})` : ''}
        </button>
      </div>

      {showFilters && (
        <div className="mb-6 p-4 bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-card flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-400">
            Architecture
            <select
              aria-label="Architecture"
              value={filters.encoder_type ?? ''}
              onChange={(e) => set({ encoder_type: e.target.value || undefined })}
              className={FIELD}
            >
              <option value="">Any architecture</option>
              {ARCHITECTURE_FILTER_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-400">
            Created from
            <input
              type="date"
              aria-label="Created from"
              value={filters.created_from ?? ''}
              max={filters.created_to || undefined}
              onChange={(e) => set({ created_from: e.target.value || undefined })}
              className={FIELD}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-400">
            Created to
            <input
              type="date"
              aria-label="Created to"
              value={filters.created_to ?? ''}
              min={filters.created_from || undefined}
              onChange={(e) => set({ created_to: e.target.value || undefined })}
              className={FIELD}
            />
          </label>
          {active && (
            <button
              onClick={clearAll}
              className="px-3 py-2 text-sm rounded-control bg-white dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300"
            >
              Clear filters
            </button>
          )}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader className="w-8 h-8 text-slate-600 dark:text-slate-400 animate-spin" />
          <span className="ml-3 text-slate-600 dark:text-slate-400">Loading templates...</span>
        </div>
      ) : templates.length === 0 ? (
        <div className="text-center py-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white dark:bg-slate-800 mb-4">
            <Search className="w-8 h-8 text-slate-500" />
          </div>
          <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">
            No templates match these filters
          </h3>
          <button
            onClick={clearAll}
            className="mt-4 px-4 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-control transition-colors text-slate-700 dark:text-slate-300"
          >
            Clear filters
          </button>
        </div>
      ) : (
      <>
      {/* Results Count */}
      <div className="mb-4 text-sm text-slate-600 dark:text-slate-400">
        Showing {templates.length} of {total} template{total !== 1 ? 's' : ''}
        {active && ' (filtered)'}
      </div>

      {/* Template Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        {templates.map((template) => (
          <TrainingTemplateCard
            key={template.id}
            template={template}
            onClick={onTemplateClick ? () => onTemplateClick(template) : undefined}
            onEdit={onEdit}
            onDelete={onDelete}
            onToggleFavorite={onToggleFavorite}
            onDuplicate={onDuplicate}
          />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && onPageChange && (
        <div className="flex items-center justify-between mt-6 pt-6 border-t border-slate-200 dark:border-slate-800">
          <div className="text-sm text-slate-600 dark:text-slate-400">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={!hasPrev}
              className="flex items-center gap-1 px-3 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:bg-slate-50 dark:disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 disabled:text-slate-600 rounded-control transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={!hasNext}
              className="flex items-center gap-1 px-3 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:bg-slate-50 dark:disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 disabled:text-slate-600 rounded-control transition-colors"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
      </>
      )}
    </div>
  );
}
