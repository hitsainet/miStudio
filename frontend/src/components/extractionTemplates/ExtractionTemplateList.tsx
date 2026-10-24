/**
 * ExtractionTemplateList component for displaying a list of extraction templates.
 *
 * Renders a grid of template cards with search, filters (hook type and sort
 * order) and pagination. Filters are CONTROLLED by the panel and sent to
 * GET /extraction-templates: the list is paginated server-side, so a
 * browser-side filter or sort would act only on the page in hand.
 */

import { useEffect, useRef, useState } from 'react';
import { Search, SlidersHorizontal, Loader } from 'lucide-react';
import { ExtractionHookType, ExtractionTemplate } from '../../types/extractionTemplate';
import { ExtractionTemplateCard } from './ExtractionTemplateCard';
import {
  EXTRACTION_SORTS,
  ExtractionSortKey,
  ExtractionTemplateFilters,
  extractionFiltersActive,
} from '../../utils/templateListFilters';

export const HOOK_TYPE_FILTER_OPTIONS: { value: ExtractionHookType; label: string }[] = [
  { value: 'residual', label: 'Residual' },
  { value: 'mlp', label: 'MLP' },
  { value: 'attention', label: 'Attention' },
];

/** Debounce for the search box, so typing sends one request, not one per key. */
export const SEARCH_DEBOUNCE_MS = 300;

const FIELD = 'px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-sm text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-2 focus:ring-emerald-500';

interface ExtractionTemplateListProps {
  templates: ExtractionTemplate[];
  loading?: boolean;
  onTemplateClick?: (template: ExtractionTemplate) => void;
  onEdit?: (template: ExtractionTemplate) => void;
  onDelete?: (id: string) => void;
  onToggleFavorite?: (id: string) => void;
  onDuplicate?: (template: ExtractionTemplate) => void;
  /** Active filters (controlled) and their setter. */
  filters?: ExtractionTemplateFilters;
  onFiltersChange?: (filters: ExtractionTemplateFilters) => void;
  /** Server-side total across all pages, for the results count. */
  totalCount?: number;
  // Pagination
  currentPage?: number;
  totalPages?: number;
  hasNext?: boolean;
  hasPrev?: boolean;
  onPageChange?: (page: number) => void;
}

export function ExtractionTemplateList({
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
}: ExtractionTemplateListProps) {
  const [searchText, setSearchText] = useState(filters.search ?? '');
  const [showFilters, setShowFilters] = useState(
    !!(filters.hook_type || (filters.sort && filters.sort !== 'newest')));
  const active = extractionFiltersActive(filters);

  const latest = useRef({ filters, onFiltersChange });
  latest.current = { filters, onFiltersChange };

  useEffect(() => {
    if ((latest.current.filters.search ?? '') === searchText) return;
    const t = setTimeout(() => {
      latest.current.onFiltersChange?.({ ...latest.current.filters, search: searchText });
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [searchText]);

  const set = (patch: Partial<ExtractionTemplateFilters>) =>
    onFiltersChange?.({ ...filters, search: searchText, ...patch });

  const clearAll = () => {
    setSearchText('');
    onFiltersChange?.({ sort: filters.sort });
  };

  if (!loading && templates.length === 0 && !active) {
    return (
      <div className="text-center py-12">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-white dark:bg-slate-800 mb-4">
          <Search className="w-8 h-8 text-slate-500" />
        </div>
        <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-2">No templates found</h3>
        <p className="text-sm text-slate-500">
          Create your first extraction template to get started.
        </p>
      </div>
    );
  }

  const total = totalCount ?? templates.length;

  return (
    <div className="space-y-4">
      {/* Search Bar — always mounted, so typing keeps focus across reloads */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            aria-label="Search templates"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search templates by name or description..."
            className="w-full pl-10 pr-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          aria-expanded={showFilters}
          className={`p-2 rounded transition-colors ${
            showFilters
              ? 'bg-emerald-600 text-white'
              : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
          }`}
          title="Toggle filters"
          aria-label="Toggle filters"
        >
          <SlidersHorizontal className="w-5 h-5" />
        </button>
      </div>

      {showFilters && (
        <div className="p-4 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm flex flex-wrap items-end gap-4">
          <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-400">
            Hook type
            <select
              aria-label="Hook type"
              value={filters.hook_type ?? ''}
              onChange={(e) => set({ hook_type: (e.target.value || undefined) as ExtractionHookType | undefined })}
              className={FIELD}
            >
              <option value="">Any hook type</option>
              {HOOK_TYPE_FILTER_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-400">
            Sort
            <select
              aria-label="Sort"
              value={filters.sort ?? 'newest'}
              onChange={(e) => set({ sort: e.target.value as ExtractionSortKey })}
              className={FIELD}
            >
              {(Object.keys(EXTRACTION_SORTS) as ExtractionSortKey[]).map((k) => (
                <option key={k} value={k}>{EXTRACTION_SORTS[k].label}</option>
              ))}
            </select>
          </label>
          {active && (
            <button
              onClick={clearAll}
              className="px-3 py-2 text-sm rounded-sm bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300"
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
            className="mt-4 px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-600 text-slate-700 dark:text-slate-300 rounded-sm transition-colors"
          >
            Clear filters
          </button>
        </div>
      ) : (
      <>
      {/* Results Count */}
      <div className="text-sm text-slate-600 dark:text-slate-400">
        Showing {templates.length} of {total} template{total !== 1 ? 's' : ''}
        {active && ' (filtered)'}
      </div>

      {/* Template Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {templates.map((template) => (
          <ExtractionTemplateCard
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
        <div className="flex items-center justify-center gap-4 pt-6">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={!hasPrev}
            className="px-4 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 disabled:text-slate-600 rounded-sm transition-colors"
          >
            Previous
          </button>
          <span className="text-sm text-slate-600 dark:text-slate-400">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={!hasNext}
            className="px-4 py-2 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:bg-slate-900 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 disabled:text-slate-600 rounded-sm transition-colors"
          >
            Next
          </button>
        </div>
      )}
      </>
      )}
    </div>
  );
}
