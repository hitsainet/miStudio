# Enhanced Dataset Tokenization Progress Display - Implementation Tasks

## Overview
Enhance the dataset tile (DatasetCard.tsx) to display rich, real-time progress information during tokenization jobs, replacing the simple progress bar with detailed statistics, timing estimates, and filter metrics.

## Approved Features
-  Progress bar showing completion percentage
-  Estimated time remaining
-  Current stage indicator (e.g., "Tokenizing...", "Filtering...", "Saving...")
-  Filter statistics (samples processed/kept/filtered, filter mode)
-  Tokens per second processing rate
-  Elapsed time
-  Quick cancel button
- L Memory usage (excluded - redundant with System Monitor tab)

## Current State Analysis

### Frontend Components
- **DatasetCard.tsx** (lines 135-139): Simple progress bar, no detailed stats
- **ProgressBar.tsx**: Basic component showing percentage only
- **datasetsStore.ts**: Store only tracks progress percentage via `updateDatasetProgress()`

### Backend Progress Emission
- **dataset_tasks.py** (lines 400-800): Tokenization task with stage-based progress updates
- **websocket_emitter.py** (lines 121-156): `emit_dataset_progress()` function for WebSocket emission
- **Current Progress Stages**:
  - 10%: Loading tokenizer
  - 20%: Loading dataset
  - 30%: Analyzing schema
  - 40%: Tokenizing (main processing loop)
  - 90%: Saving
  - 100%: Complete

### Data Flow
1. Worker task updates progress via `emit_dataset_progress()`
2. WebSocket emits to channel: `datasets/{dataset_id}/progress`
3. Event: `dataset:progress`
4. Frontend store receives via `updateDatasetProgress(id, progress)`
5. DatasetCard component displays progress bar

## Implementation Tasks

### Phase 1: Backend Enhancements (Emit Additional Statistics)

#### Task 1.1: Enhance Tokenization Progress Payload
**File**: `/home/x-sean/app/miStudio/backend/src/workers/dataset_tasks.py`

**Changes Needed**:
- [ ] Add tracking variables at task start:
  ```python
  start_time = datetime.now(UTC)
  samples_processed = 0
  samples_kept = 0
  samples_filtered = 0
  tokens_processed = 0
  batch_count = 0
  ```
- [ ] Update progress emission payload to include:
  - `stage`: Current stage name ("Loading Tokenizer", "Tokenizing", "Filtering", "Saving")
  - `samples_processed`: Total samples processed so far
  - `samples_kept`: Samples that passed filtering
  - `samples_filtered`: Samples rejected by filtering
  - `filter_mode`: Current filter mode ("strict", "conservative", etc.)
  - `filter_threshold`: Junk ratio threshold (e.g., 0.5)
  - `tokens_processed`: Total tokens generated
  - `elapsed_seconds`: Seconds since task start
  - `tokens_per_second`: Processing rate
  - `estimated_seconds_remaining`: Time estimate based on current rate

**Emission Points** (modify existing `emit_dataset_progress()` calls):
1. **Line ~540** - Loading tokenizer stage:
   ```python
   emit_dataset_progress(
       dataset_id=str(dataset_uuid),
       event="progress",
       data={
           "dataset_id": str(dataset_uuid),
           "progress": 10.0,
           "stage": "Loading Tokenizer",
           "elapsed_seconds": (datetime.now(UTC) - start_time).total_seconds(),
       }
   )
   ```

2. **Line ~555** - Loading dataset stage:
   ```python
   emit_dataset_progress(
       dataset_id=str(dataset_uuid),
       event="progress",
       data={
           "dataset_id": str(dataset_uuid),
           "progress": 20.0,
           "stage": "Loading Dataset",
           "elapsed_seconds": (datetime.now(UTC) - start_time).total_seconds(),
       }
   )
   ```

3. **Line ~585** - Analyzing schema stage:
   ```python
   emit_dataset_progress(
       dataset_id=str(dataset_uuid),
       event="progress",
       data={
           "dataset_id": str(dataset_uuid),
           "progress": 30.0,
           "stage": "Analyzing Schema",
           "elapsed_seconds": (datetime.now(UTC) - start_time).total_seconds(),
       }
   )
   ```

4. **Main tokenization loop** (~line 650-750, inside batch processing loop):
   - [ ] Find the tokenization loop with batch processing
   - [ ] Track samples and tokens per batch
   - [ ] Emit progress every N batches (e.g., every 10 batches or every 1000 samples)
   - [ ] Calculate tokens per second: `tokens_processed / elapsed_seconds`
   - [ ] Estimate remaining time: `(total_samples - samples_processed) / (samples_processed / elapsed_seconds)`
   - [ ] Include filter statistics in payload:
   ```python
   elapsed = (datetime.now(UTC) - start_time).total_seconds()
   tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
   samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
   remaining_samples = total_samples - samples_processed
   estimated_remaining = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0

   emit_dataset_progress(
       dataset_id=str(dataset_uuid),
       event="progress",
       data={
           "dataset_id": str(dataset_uuid),
           "progress": current_progress,
           "stage": "Tokenizing",
           "samples_processed": samples_processed,
           "samples_kept": samples_kept,
           "samples_filtered": samples_filtered,
           "filter_mode": filter_mode,
           "filter_threshold": filter_threshold,
           "tokens_processed": tokens_processed,
           "elapsed_seconds": elapsed,
           "tokens_per_second": tokens_per_sec,
           "estimated_seconds_remaining": estimated_remaining,
       }
   )
   ```

5. **Saving stage** (~line 750-800):
   ```python
   emit_dataset_progress(
       dataset_id=str(dataset_uuid),
       event="progress",
       data={
           "dataset_id": str(dataset_uuid),
           "progress": 90.0,
           "stage": "Saving",
           "samples_processed": total_samples,
           "samples_kept": samples_kept,
           "samples_filtered": samples_filtered,
           "elapsed_seconds": (datetime.now(UTC) - start_time).total_seconds(),
       }
   )
   ```

**Acceptance Criteria**:
- All progress emissions include `stage` field
- Tokenization stage includes complete statistics
- Filtering statistics accurate (samples_kept + samples_filtered = samples_processed)
- Time calculations accurate (no division by zero)
- Emissions don't spam (rate-limited to reasonable frequency, e.g., every 1000 samples)

---

#### Task 1.2: Add Completion Timestamp to Final Emission
**File**: `/home/x-sean/app/miStudio/backend/src/workers/dataset_tasks.py`

**Changes Needed**:
- [ ] At task completion (line ~800-850), emit final progress with:
  ```python
  completed_at = datetime.now(UTC)
  total_elapsed = (completed_at - start_time).total_seconds()

  emit_dataset_progress(
      dataset_id=str(dataset_uuid),
      event="completed",
      data={
          "dataset_id": str(dataset_uuid),
          "progress": 100.0,
          "stage": "Complete",
          "samples_processed": total_samples,
          "samples_kept": samples_kept,
          "samples_filtered": samples_filtered,
          "filter_mode": filter_mode,
          "filter_threshold": filter_threshold,
          "tokens_processed": total_tokens,
          "elapsed_seconds": total_elapsed,
          "completed_at": completed_at.isoformat(),
      }
  )
  ```

**Acceptance Criteria**:
- Completion event includes final statistics
- `completed_at` timestamp in ISO format
- Total elapsed time accurate

---

### Phase 2: Frontend Type Definitions

#### Task 2.1: Extend Dataset Type with Progress Details
**File**: `/home/x-sean/app/miStudio/frontend/src/types/dataset.ts`

**Changes Needed**:
- [ ] Add new interface for detailed progress:
  ```typescript
  export interface DatasetTokenizationProgress {
    stage?: string;                    // "Loading Tokenizer", "Tokenizing", "Filtering", "Saving"
    samples_processed?: number;        // Total samples processed
    samples_kept?: number;             // Samples that passed filtering
    samples_filtered?: number;         // Samples rejected by filtering
    filter_mode?: string;              // "strict", "conservative", etc.
    filter_threshold?: number;         // Junk ratio threshold (0.0-1.0)
    tokens_processed?: number;         // Total tokens generated
    elapsed_seconds?: number;          // Seconds since start
    tokens_per_second?: number;        // Processing rate
    estimated_seconds_remaining?: number; // Time estimate
    completed_at?: string;             // ISO timestamp when completed
  }
  ```

- [ ] Update Dataset interface (line 121-142):
  ```typescript
  export interface Dataset {
    // ... existing fields ...

    /** Detailed tokenization progress information */
    tokenization_progress?: DatasetTokenizationProgress;
  }
  ```

**Acceptance Criteria**:
- New interface matches backend payload structure
- All fields optional (for backward compatibility)
- TypeScript compilation succeeds

---

### Phase 3: Frontend Store Updates

#### Task 3.1: Enhance Store to Track Detailed Progress
**File**: `/home/x-sean/app/miStudio/frontend/src/stores/datasetsStore.ts`

**Changes Needed**:
- [ ] Update `updateDatasetProgress` function (line 216-224) to accept full progress object:
  ```typescript
  // Update dataset progress (called by WebSocket)
  updateDatasetProgress: (id: string, progress: number, progressDetails?: DatasetTokenizationProgress) => {
    set((state) => ({
      datasets: state.datasets.map((dataset) =>
        dataset.id === id
          ? {
              ...dataset,
              progress,
              tokenization_progress: progressDetails
            }
          : dataset
      ),
    }));
  },
  ```

- [ ] Update store interface (line 38):
  ```typescript
  updateDatasetProgress: (id: string, progress: number, progressDetails?: DatasetTokenizationProgress) => void;
  ```

**Acceptance Criteria**:
- Store accepts optional progress details
- Progress details stored alongside simple percentage
- Backward compatible (works with or without details)

---

#### Task 3.2: Update WebSocket Handler to Parse Progress Details
**File**: `/home/x-sean/app/miStudio/frontend/src/contexts/WebSocketContext.tsx` (if exists) or the component that handles WebSocket events

**Changes Needed**:
- [ ] Find the WebSocket event handler for `dataset:progress`
- [ ] Update to extract progress details from payload:
  ```typescript
  socket.on('dataset:progress', (data: any) => {
    const { dataset_id, progress, ...progressDetails } = data;

    // Update store with both progress percentage and details
    useDatasetsStore.getState().updateDatasetProgress(
      dataset_id,
      progress,
      progressDetails  // Pass through all additional fields
    );
  });
  ```

**Acceptance Criteria**:
- WebSocket handler extracts all progress fields
- Progress details passed to store
- No console errors on receiving new payload format

---

### Phase 4: UI Components

#### Task 4.1: Create Enhanced Tokenization Progress Component
**File**: `/home/x-sean/app/miStudio/frontend/src/components/datasets/TokenizationProgress.tsx` (NEW FILE)

**Changes Needed**:
- [ ] Create new component to display detailed progress:
  ```typescript
  import { Clock, Zap, Filter, AlertCircle } from 'lucide-react';
  import { DatasetTokenizationProgress } from '../../types/dataset';
  import { ProgressBar } from '../common/ProgressBar';

  interface TokenizationProgressProps {
    progress: number;
    details?: DatasetTokenizationProgress;
    onCancel?: () => void;
  }

  export function TokenizationProgress({
    progress,
    details,
    onCancel
  }: TokenizationProgressProps) {
    // Calculate derived values
    const filterPercent = details?.samples_processed
      ? (details.samples_filtered / details.samples_processed * 100).toFixed(1)
      : '0';
    const keepPercent = details?.samples_processed
      ? (details.samples_kept / details.samples_processed * 100).toFixed(1)
      : '0';

    // Format time duration
    const formatDuration = (seconds?: number): string => {
      if (!seconds) return '--:--';
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Format number with commas
    const formatNumber = (num?: number): string => {
      if (num === undefined || num === null) return '0';
      return num.toLocaleString();
    };

    return (
      <div className="mt-3 space-y-2">
        {/* Progress bar */}
        <ProgressBar progress={progress} showPercentage={true} />

        {/* Stage indicator */}
        {details?.stage && (
          <div className="text-xs text-slate-400 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span>{details.stage}</span>
          </div>
        )}

        {/* Timing information - 2 column grid */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          {/* Elapsed time */}
          <div className="flex items-center gap-1 text-slate-400">
            <Clock className="w-3 h-3" />
            <span>Elapsed: {formatDuration(details?.elapsed_seconds)}</span>
          </div>

          {/* Estimated remaining */}
          {details?.estimated_seconds_remaining !== undefined && (
            <div className="flex items-center gap-1 text-slate-400">
              <Clock className="w-3 h-3" />
              <span>Remaining: {formatDuration(details.estimated_seconds_remaining)}</span>
            </div>
          )}
        </div>

        {/* Processing rate */}
        {details?.tokens_per_second !== undefined && (
          <div className="text-xs text-slate-400 flex items-center gap-1">
            <Zap className="w-3 h-3" />
            <span>{Math.round(details.tokens_per_second).toLocaleString()} tokens/sec</span>
          </div>
        )}

        {/* Filter statistics - only show if filtering is active */}
        {details?.samples_processed !== undefined && details.samples_processed > 0 && (
          <div className="space-y-1">
            <div className="text-xs text-slate-400 flex items-center gap-1">
              <Filter className="w-3 h-3" />
              <span>
                Filter: {details.filter_mode || 'N/A'}
                {details.filter_threshold && ` (${(details.filter_threshold * 100).toFixed(0)}% threshold)`}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-xs">
              {/* Samples processed */}
              <div className="text-slate-400">
                <div className="font-medium">Processed</div>
                <div>{formatNumber(details.samples_processed)}</div>
              </div>

              {/* Samples kept */}
              <div className="text-emerald-400">
                <div className="font-medium">Kept</div>
                <div>{formatNumber(details.samples_kept)} ({keepPercent}%)</div>
              </div>

              {/* Samples filtered */}
              <div className="text-amber-400">
                <div className="font-medium">Filtered</div>
                <div>{formatNumber(details.samples_filtered)} ({filterPercent}%)</div>
              </div>
            </div>
          </div>
        )}

        {/* Cancel button */}
        {onCancel && (
          <button
            onClick={onCancel}
            className="w-full mt-2 px-3 py-1.5 text-xs bg-red-600/10 hover:bg-red-600/20 text-red-400 rounded transition-colors flex items-center justify-center gap-1"
          >
            <AlertCircle className="w-3 h-3" />
            Cancel Tokenization
          </button>
        )}
      </div>
    );
  }
  ```

**Acceptance Criteria**:
- Component displays all approved features
- Gracefully handles missing data (shows '--' or 'N/A')
- Responsive layout works on different screen sizes
- Icons from lucide-react
- Tailwind classes match Mock UI style (slate dark theme)
- Cancel button styled as destructive action

---

#### Task 4.2: Update DatasetCard to Use Enhanced Progress Component
**File**: `/home/x-sean/app/miStudio/frontend/src/components/datasets/DatasetCard.tsx`

**Changes Needed**:
- [ ] Import new component:
  ```typescript
  import { TokenizationProgress } from './TokenizationProgress';
  ```

- [ ] Replace simple progress bar (lines 135-139) with:
  ```typescript
  {showProgress && dataset.progress !== undefined && (
    <TokenizationProgress
      progress={dataset.progress}
      details={dataset.tokenization_progress}
      onCancel={onCancel ? () => onCancel(dataset.id) : undefined}
    />
  )}
  ```

- [ ] Ensure DatasetCard accepts `onCancel` prop if not already:
  ```typescript
  interface DatasetCardProps {
    dataset: Dataset;
    onClick?: () => void;
    onDelete?: (id: string) => void;
    onCancel?: (id: string) => void;  // Add if missing
  }
  ```

**Acceptance Criteria**:
- Enhanced progress replaces simple progress bar
- Card layout not broken by new component
- Cancel functionality wired up
- TypeScript compilation succeeds

---

#### Task 4.3: Wire Up Cancel Functionality in DatasetsPanel
**File**: `/home/x-sean/app/miStudio/frontend/src/components/panels/DatasetsPanel.tsx`

**Changes Needed**:
- [ ] Ensure `cancelDownload` is passed to DatasetCard:
  ```typescript
  <DatasetCard
    key={dataset.id}
    dataset={dataset}
    onClick={() => handleDatasetClick(dataset)}
    onDelete={handleDelete}
    onCancel={handleCancelDownload}  // Add if missing
  />
  ```

- [ ] Ensure `handleCancelDownload` exists and calls store:
  ```typescript
  const handleCancelDownload = async (id: string) => {
    try {
      await cancelDownload(id);
    } catch (error) {
      console.error('Failed to cancel:', error);
    }
  };
  ```

**Acceptance Criteria**:
- Cancel button triggers `cancelDownload` API call
- Dataset status updates to ERROR after cancellation
- User sees cancellation confirmation (or error message)

---

### Phase 5: Testing & Refinement

#### Task 5.1: Manual Testing Checklist
- [ ] Start a new tokenization job with filtering enabled
- [ ] Verify enhanced progress displays all statistics:
  - [ ] Progress bar with percentage
  - [ ] Current stage indicator
  - [ ] Elapsed time
  - [ ] Estimated remaining time (after processing starts)
  - [ ] Tokens per second rate
  - [ ] Filter statistics (processed/kept/filtered counts and percentages)
  - [ ] Filter mode and threshold
- [ ] Verify cancel button appears and works
- [ ] Test with different filter modes (minimal, conservative, strict)
- [ ] Test with filtering disabled (ensure no filter stats shown)
- [ ] Verify completion shows final statistics
- [ ] Check backward compatibility (old datasets without progress details)

---

#### Task 5.2: Performance & UX Refinements
- [ ] Verify WebSocket emission rate is reasonable (not spamming)
- [ ] Confirm UI doesn't flicker or jump during updates
- [ ] Check that long numbers format nicely with commas
- [ ] Verify time formatting is human-readable
- [ ] Test on different screen sizes (responsive layout)
- [ ] Ensure cancel button has proper affordance (red, destructive style)

---

#### Task 5.3: Error Handling
- [ ] Handle missing or incomplete progress data gracefully
- [ ] Show fallback UI if WebSocket disconnects mid-progress
- [ ] Display error message if cancellation fails
- [ ] Verify error states don't break the UI

---

## File Modifications Summary

### Backend Files
1. `/home/x-sean/app/miStudio/backend/src/workers/dataset_tasks.py`
   - Add tracking variables for statistics
   - Enhance all `emit_dataset_progress()` calls with detailed payload
   - Add rate limiting to prevent emission spam

### Frontend Files (New)
1. `/home/x-sean/app/miStudio/frontend/src/components/datasets/TokenizationProgress.tsx`
   - New component for enhanced progress display

### Frontend Files (Modified)
1. `/home/x-sean/app/miStudio/frontend/src/types/dataset.ts`
   - Add `DatasetTokenizationProgress` interface
   - Update `Dataset` interface

2. `/home/x-sean/app/miStudio/frontend/src/stores/datasetsStore.ts`
   - Update `updateDatasetProgress` signature
   - Store detailed progress information

3. `/home/x-sean/app/miStudio/frontend/src/components/datasets/DatasetCard.tsx`
   - Replace simple ProgressBar with TokenizationProgress component
   - Add `onCancel` prop

4. `/home/x-sean/app/miStudio/frontend/src/components/panels/DatasetsPanel.tsx`
   - Wire up cancel functionality to DatasetCard

5. WebSocket handler (location TBD - find during implementation)
   - Update event handler to extract and pass progress details

---

## Implementation Order

### Recommended Sequence:
1. **Phase 2**: Frontend types (foundational, no runtime impact)
2. **Phase 1**: Backend enhancements (start emitting new data)
3. **Phase 3**: Frontend store (receive and store new data)
4. **Phase 4**: UI components (display new data)
5. **Phase 5**: Testing and refinement

### Rationale:
- Types first allows TypeScript to guide implementation
- Backend changes emit data that frontend initially ignores (safe)
- Store updates capture data but don't change UI yet (safe)
- UI updates are the final, user-facing changes
- Testing validates the complete flow

---

## Success Criteria

### Definition of Done:
-  All approved features implemented and visible during tokenization
-  Filter statistics accurate and match backend calculations
-  Time estimates reasonable and update in real-time
-  Cancel button functional and provides feedback
-  No console errors or warnings
-  Backward compatible with existing datasets
-  Performance acceptable (no UI lag from frequent updates)
-  Code follows project standards (TypeScript strict, component patterns)
-  Manual testing checklist completed
-  No regressions in existing dataset functionality

---

## Estimated Effort

### Time Estimates:
- **Phase 1 (Backend)**: 3-4 hours
  - Tracking logic: 1 hour
  - Emission points: 1.5 hours
  - Testing/debugging: 1.5 hours

- **Phase 2 (Types)**: 30 minutes
  - Interface definitions: 30 minutes

- **Phase 3 (Store)**: 1 hour
  - Store updates: 30 minutes
  - WebSocket handler: 30 minutes

- **Phase 4 (UI)**: 3-4 hours
  - TokenizationProgress component: 2 hours
  - Integration with DatasetCard: 30 minutes
  - Cancel wiring: 30 minutes
  - Styling/polish: 1 hour

- **Phase 5 (Testing)**: 2-3 hours
  - Manual testing: 1.5 hours
  - Bug fixes: 1 hour
  - Performance tuning: 30 minutes

**Total Estimated Time**: 10-13 hours

---

## Notes

### Design Decisions:
1. **Separate Component**: Created `TokenizationProgress` as standalone component for reusability and maintainability
2. **Optional Data**: All progress details optional to ensure backward compatibility
3. **Rate Limiting**: Backend should emit progress updates at reasonable intervals (e.g., every 1000 samples) to avoid WebSocket spam
4. **Time Formatting**: Using MM:SS format for durations (familiar and concise)
5. **Filter Stats Layout**: 3-column grid for processed/kept/filtered provides clear visual comparison
6. **Cancel Button**: Placed at bottom of progress section with destructive styling (red) to prevent accidental clicks

### Future Enhancements (Not in Scope):
- Pause/resume functionality
- Historical progress chart (sparkline)
- Notification on completion
- Progress persistence across page refreshes
- Multiple tokenization jobs in parallel

---

## References
- Mock UI specification: `/home/x-sean/app/miStudio/0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
- WebSocket implementation guide: `/home/x-sean/app/miStudio/CLAUDE.md` (Real-time Updates Architecture section)
- Existing progress pattern: System Monitor WebSocket migration (HP-1)
