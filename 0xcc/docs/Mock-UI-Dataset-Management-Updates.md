# Mock UI Dataset Management Production Alignment

## Summary
This document details the updates made to align the Mock UI reference implementation (`Mock-embedded-interp-ui.tsx`) with the production Dataset Management implementation completed in Phase 6.

## Date
2025-10-11

## Status
✅ **COMPLETED** - Dataset Management section is now production-aligned

## Changes Made

### 1. TypeScript Interfaces (Lines 311-379) ✅

**Updated Dataset Interface** - Now includes all production fields:
- Added `hf_repo_id?: string` - HuggingFace repository ID
- Changed `status` from `'downloading' | 'ingesting' | 'ready' | 'error'` to `'downloading' | 'processing' | 'ready' | 'error'`
- Changed `error?: string` to `error_message?: string`
- Added `raw_path?: string` - File path to raw dataset
- Added `tokenized_path?: string` - File path to tokenized dataset
- Added `num_samples?: number` - Total number of samples
- Changed `size: string` to `size_bytes?: number` - Size in bytes
- Added `metadata?: DatasetMetadata` - Structured metadata
- Added `created_at: string` and `updated_at: string` - ISO timestamps

**New Metadata Interfaces**:
```typescript
interface SchemaMetadata {
  text_columns: string[];
  column_info: Record<string, string>;
  all_columns: string[];
  is_multi_column: boolean;
}

interface TokenizationMetadata {
  tokenizer_name: string;
  text_column_used: string;
  max_length: number;
  stride: number;
  num_tokens: number;
  avg_seq_length: number;
  min_seq_length: number;
  max_seq_length: number;
}

interface DownloadMetadata {
  split?: string;
  config?: string;
  access_token_provided?: boolean;
}

interface DatasetMetadata {
  schema?: SchemaMetadata;
  tokenization?: TokenizationMetadata;
  download?: DownloadMetadata;
}
```

**Updated API Contract Documentation**:
- Changed from `/api/datasets` to `/api/v1/datasets`
- Added `/api/v1/datasets/:id/tokenize` endpoint
- Added `/api/v1/datasets/:id/samples?page=1&limit=20` endpoint
- Updated status transitions: `downloading -> processing -> ready`
- Updated real-time communication: WebSocket events on `datasets/{id}/progress` channel
- Events: `'progress'`, `'completed'`, `'error'`

### 2. Icon Imports (Line 2) ✅

Added missing icons:
```typescript
import { ..., FileText, BarChart, Settings, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';
```

### 3. DatasetsPanel Component (Lines 1352-1500) ✅

**Structure Changes**:
- Added proper page wrapper with `min-h-screen bg-slate-950`
- Added `max-w-7xl mx-auto px-6 py-8` container
- Updated header: `<h1>` with title, `<p>` with description
- Added `split` and `config` state variables

**Header Section** (PRODUCTION-ALIGNED):
```typescript
<div className="mb-8">
  <h1 className="text-2xl font-semibold text-slate-100 mb-2">Datasets</h1>
  <p className="text-slate-400">
    Manage training datasets from HuggingFace or local sources
  </p>
</div>
```

**Download Form Updates**:
- Changed label from "Dataset Repository" to "Repository ID"
- Updated placeholder from "e.g., roneneldan/TinyStories" to "username/dataset-name"
- Changed placeholder for token from "hf_xxxxxxxxxxxxxxxxxxxx" to "hf_..."
- Added 2-column grid for split and config fields:
  ```typescript
  <div className="grid grid-cols-2 gap-4">
    <div>
      <label>Split (optional)</label>
      <input placeholder="train, validation, test" />
      <p className="text-xs">Dataset split to download</p>
    </div>
    <div>
      <label>Config (optional)</label>
      <input placeholder="en, zh, etc." />
      <p className="text-xs">Dataset configuration</p>
    </div>
  </div>
  ```
- Updated button styling and onClick to pass split and config
- Updated all input styling to use `bg-slate-800` instead of `bg-slate-900`

**Empty State** (PRODUCTION-ALIGNED):
```typescript
{datasets.length === 0 && (
  <div className="text-center py-12">
    <p className="text-slate-400 text-lg">No datasets yet</p>
    <p className="text-slate-500 mt-2">Download a dataset from HuggingFace to get started</p>
  </div>
)}
```

**Datasets Grid** (PRODUCTION-ALIGNED):
- Added section heading: "Your Datasets ({count})"
- Changed from `<div className="grid gap-4">` to `<div className="grid gap-4 md:grid-cols-2">`
- Extracted DatasetCard to separate component

### 4. DatasetCard Component (Lines 1503-1594) ✅

**New Standalone Component** (PRODUCTION-ALIGNED):
```typescript
function DatasetCard({ dataset, onClick }) {
  const isClickable = dataset.status === 'ready';
  const showProgress = dataset.status === 'downloading' || dataset.status === 'processing';
  const isTokenized = dataset.metadata?.tokenization !== undefined;

  // Status icon mapping (CheckCircle, Loader, Activity)
  const StatusIcon = dataset.status === 'ready' ? CheckCircle
    : dataset.status === 'downloading' ? Loader
    : dataset.status === 'processing' || dataset.status === 'error' ? Activity
    : Database;

  const iconClassName = dataset.status === 'downloading' ? 'animate-spin' : '';

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 transition-all ...">
      <div className="flex items-start gap-4">
        {/* Database icon with flex-shrink-0 */}
        <div className="flex-shrink-0">
          <Database className="w-8 h-8 text-slate-400" />
        </div>

        <div className="flex-1 min-w-0">
          {/* Title and status */}
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-slate-100 truncate">
                {dataset.name}
              </h3>
              <p className="text-sm text-slate-400 mt-1 truncate">
                Source: {dataset.source}
                {dataset.hf_repo_id && ` • ${dataset.hf_repo_id}`}
              </p>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <StatusIcon className={`w-5 h-5 text-slate-400 ${iconClassName}`} />
              <StatusBadge status={dataset.status} />
            </div>
          </div>

          {/* Size */}
          {dataset.size_bytes !== undefined && dataset.size_bytes > 0 && (
            <p className="text-sm text-slate-400 mt-2">
              Size: {formatBytes(dataset.size_bytes)}
            </p>
          )}

          {/* Samples */}
          {dataset.num_samples !== undefined && dataset.num_samples > 0 && (
            <p className="text-sm text-slate-400 mt-1">
              Samples: {dataset.num_samples.toLocaleString()}
            </p>
          )}

          {/* Tokenized badge */}
          {isTokenized && (
            <div className="inline-flex items-center gap-1.5 mt-2 px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-400">
              <Info className="w-3 h-3" />
              <span>Tokenized</span>
            </div>
          )}

          {/* Progress bar */}
          {showProgress && dataset.progress !== undefined && (
            <div className="mt-4">
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300"
                  style={{ width: `${dataset.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error message */}
          {dataset.error_message && (
            <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
              {dataset.error_message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Key Features**:
- Proper flex layout with `flex-shrink-0` on icon
- Status icon changes based on dataset status (ready, downloading, processing, error)
- Animated spinner for downloading status
- Shows size in bytes using `formatBytes()` helper
- Shows tokenized badge when `metadata.tokenization` exists
- Progress bar with gradient `from-blue-500 to-blue-400`
- Error message display

### 5. StatusBadge Component (Lines 1596-1613) ✅

**New Standalone Component** (PRODUCTION-ALIGNED):
```typescript
function StatusBadge({ status }) {
  const statusColors = {
    downloading: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    processing: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    ready: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    error: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  const normalizedStatus = String(status).toLowerCase();
  const colorClass = statusColors[normalizedStatus] || statusColors.ready;

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${colorClass}`}>
      {normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1)}
    </span>
  );
}
```

**Badge Colors**:
- `downloading`: Blue (`bg-blue-500/20 text-blue-400 border-blue-500/30`)
- `processing`: Yellow (`bg-yellow-500/20 text-yellow-400 border-yellow-500/30`)
- `ready`: Emerald (`bg-emerald-500/20 text-emerald-400 border-emerald-500/30`)
- `error`: Red (`bg-red-500/20 text-red-400 border-red-500/30`)

### 6. Helper Function (Lines 1615-1622) ✅

**New formatBytes Function** (PRODUCTION-ALIGNED):
```typescript
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
```

Converts bytes to human-readable format (e.g., `1536000` → `1.46 MB`)

## DatasetDetailModal - Remaining Updates Needed

The following sections in the Mock UI still need to be updated to match production:

### 7. DatasetDetailModal Component (Lines ~3820-3887)

**Current Issues**:
- Tab icons not shown
- Header doesn't match production layout
- Missing StatusBadge in header
- Tabs should have icons (FileText, BarChart, Settings)

**Production-Aligned Updates Needed**:
```typescript
function DatasetDetailModal({ dataset, onClose }) {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: FileText },
    { id: 'samples', label: 'Samples', icon: FileText },
    { id: 'tokenization', label: 'Tokenization', icon: Settings },
    { id: 'statistics', label: 'Statistics', icon: BarChart },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-6xl w-full mx-4 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex-1 min-w-0">
            <h2 className="text-2xl font-semibold text-slate-100 truncate">
              {dataset.name}
            </h2>
            <div className="flex items-center gap-4 mt-2">
              <StatusBadge status={dataset.status} />
              <span className="text-sm text-slate-400">
                Source: {dataset.source}
              </span>
              {dataset.num_samples && (
                <span className="text-sm text-slate-400">
                  {dataset.num_samples.toLocaleString()} samples
                </span>
              )}
              {dataset.size_bytes && (
                <span className="text-sm text-slate-400">
                  {formatBytes(dataset.size_bytes)}
                </span>
              )}
            </div>
          </div>
          <button onClick={onClose} className="flex-shrink-0 ml-4 p-2 hover:bg-slate-800 rounded transition-colors">
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-800">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors ${
                  isActive
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'overview' && <OverviewTab dataset={dataset} />}
          {activeTab === 'samples' && <SamplesTab dataset={dataset} />}
          {activeTab === 'tokenization' && <TokenizationTab dataset={dataset} />}
          {activeTab === 'statistics' && <StatisticsTab dataset={dataset} />}
        </div>
      </div>
    </div>
  );
}
```

### 8. SamplesTab Component

**Updates Needed**:
- Remove search and filter UI (production doesn't have this)
- Change to pagination with 20 samples per page
- Add ChevronLeft/ChevronRight pagination controls
- Fetch samples from API: `GET /api/v1/datasets/${id}/samples?page=${page}&limit=20`
- Display loading state
- Display "Dataset not ready" empty state if status !== 'ready'
- Show sample data as key-value pairs from `sample.data` object

**Production-Aligned Implementation**:
```typescript
function SamplesTab({ dataset }) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState(null);
  const limit = 20;

  useEffect(() => {
    // Fetch samples from API
    // API: GET /api/v1/datasets/${dataset.id}/samples?page=${page}&limit=${limit}
  }, [dataset.id, page]);

  if (dataset.status !== 'ready') {
    return (
      <div className="text-center py-12">
        <FileText className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <p className="text-slate-400 text-lg">Dataset not ready</p>
        <p className="text-slate-500 mt-2">
          Samples can be viewed once the dataset is in "ready" status
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500 mb-4"></div>
        <p className="text-slate-400">Loading samples...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Pagination Header */}
      {pagination && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-slate-400">
            Showing {((page - 1) * limit) + 1} - {Math.min(page * limit, pagination.total)} of {pagination.total.toLocaleString()} samples
          </p>
          <div className="flex items-center gap-2">
            <button disabled={!pagination.has_prev} className="p-2 rounded hover:bg-slate-800 disabled:opacity-50">
              <ChevronLeft className="w-4 h-4 text-slate-400" />
            </button>
            <span className="text-sm text-slate-400">
              Page {page} of {pagination.total_pages.toLocaleString()}
            </span>
            <button disabled={!pagination.has_next} className="p-2 rounded hover:bg-slate-800 disabled:opacity-50">
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>
      )}

      {/* Samples List */}
      <div className="space-y-3">
        {samples.map((sample) => (
          <div key={sample.index} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
            <div className="flex items-start justify-between mb-2">
              <span className="text-xs font-mono text-slate-500">Sample #{sample.index}</span>
            </div>
            <div className="space-y-2">
              {Object.entries(sample.data).map(([key, value]) => (
                <div key={key}>
                  <span className="text-xs font-medium text-emerald-400">{key}:</span>
                  <pre className="text-sm text-slate-300 mt-1 whitespace-pre-wrap font-mono">
                    {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 9. TokenizationTab Component

**Updates Needed**:
- Show "Already Tokenized" info banner if `dataset.metadata.tokenization` exists
- Tokenization form with:
  - Tokenizer dropdown (gpt2, gpt2-medium, bert-base-uncased, etc.)
  - Custom tokenizer input field
  - Max length slider + number input (128-2048, step 128)
  - Stride slider + number input (0-maxLength/2, step 32)
- Submit button: "Start Tokenization" or "Re-tokenize Dataset"
- Show processing state with progress bar when `dataset.status === 'processing'`
- WebSocket progress updates
- Information panel about tokenization

**Production-Aligned Implementation**: See `/home/x-sean/app/miStudio/frontend/src/components/datasets/DatasetDetailModal.tsx` lines 486-886

### 10. StatisticsTab Component

**Updates Needed**:
- Show empty state if no tokenization metadata
- Display tokenization configuration (tokenizer, max_length, stride)
- Token statistics (total, avg, min, max)
- Sequence length distribution visualization (CSS bar chart)
- Efficiency metrics (utilization %, padding overhead %)

**Production-Aligned Implementation**: See `/home/x-sean/app/miStudio/frontend/src/components/datasets/DatasetDetailModal.tsx` lines 305-484

## Testing Checklist

- [x] Dataset interface includes all production fields
- [x] API endpoints updated to `/api/v1/*`
- [x] WebSocket event channels documented
- [x] DatasetsPanel header matches production
- [x] Download form includes split and config fields
- [x] Empty state messaging matches production
- [x] DatasetCard component extracted and styled correctly
- [x] Status badges use correct colors
- [x] Progress bars use gradient styling
- [x] formatBytes helper function added
- [ ] DatasetDetailModal header updated
- [ ] Modal tabs include icons
- [ ] SamplesTab pagination works correctly
- [ ] TokenizationTab form complete
- [ ] StatisticsTab visualization matches production

## Notes

All changes are marked with `// PRODUCTION-ALIGNED` comments in the code for easy identification.

The Mock UI now serves as an accurate reference for:
1. Backend API contracts and data structures
2. Frontend component styling and behavior
3. Real-time WebSocket communication patterns
4. User experience flows

## Next Steps

1. Complete DatasetDetailModal updates (sections 7-10)
2. Test all UI interactions match production
3. Verify all TypeScript types are correct
4. Update mock data to reflect realistic scenarios
5. Document any differences between mock and production

## Related Files

- Production Implementation: `/home/x-sean/app/miStudio/frontend/src/`
  - `components/panels/DatasetsPanel.tsx`
  - `components/datasets/DatasetCard.tsx`
  - `components/datasets/DownloadForm.tsx`
  - `components/datasets/DatasetDetailModal.tsx`
  - `types/dataset.ts`
- Mock UI Reference: `/home/x-sean/app/miStudio/0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
