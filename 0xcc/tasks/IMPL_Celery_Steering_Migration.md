# Celery-Based Steering Migration

## Overview
Migrate steering operations from synchronous FastAPI endpoints to Celery worker tasks with WebSocket progress updates. This provides process isolation, proper timeout handling via SIGKILL, and prevents zombie processes holding GPU memory.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SteeringPanel                                                      │   │
│  │    ├── User clicks "Generate"                                       │   │
│  │    ├── POST /api/v1/steering/compare → Returns { task_id }          │   │
│  │    ├── Subscribe to WebSocket: steering/{task_id}                   │   │
│  │    ├── Receive progress: { percent, message, current_strength }     │   │
│  │    └── Receive completion: { status: "completed", result: {...} }   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
┌─────────────────────────────┐    ┌─────────────────────────────┐
│      FastAPI (Lightweight)   │    │     WebSocket (Socket.IO)   │
│  ┌───────────────────────┐  │    │  ┌───────────────────────┐  │
│  │ POST /steering/compare│  │    │  │ Channel: steering/    │  │
│  │   → Validate request  │  │    │  │   {task_id}           │  │
│  │   → Submit to Celery  │  │    │  │                       │  │
│  │   → Return task_id    │  │    │  │ Events:               │  │
│  │                       │  │    │  │   - progress          │  │
│  │ GET /steering/result/ │  │    │  │   - completed         │  │
│  │   {task_id}           │  │    │  │   - failed            │  │
│  │   → Poll task status  │  │    │  │                       │  │
│  └───────────────────────┘  │    │  └───────────────────────┘  │
└─────────────────────────────┘    └─────────────────────────────┘
              │                                   ▲
              │ Redis Queue                       │ Redis PubSub
              ▼                                   │
┌─────────────────────────────────────────────────────────────────┐
│                    Celery GPU Worker                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  @celery.task(queue='steering', time_limit=180)         │   │
│  │  def steering_compare_task(request_dict):               │   │
│  │    ├── emit_progress(task_id, 0, "Loading model...")    │   │
│  │    ├── Load model + SAE to GPU                          │   │
│  │    ├── emit_progress(task_id, 20, "Generating...")      │   │
│  │    ├── For each feature/strength:                       │   │
│  │    │     ├── model.generate() [blocking OK]             │   │
│  │    │     └── emit_progress(task_id, percent, msg)       │   │
│  │    ├── emit_progress(task_id, 100, "Complete")          │   │
│  │    └── Return result dict                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Worker Config:                                                 │
│    --pool=solo --concurrency=1 --max-tasks-per-child=50        │
│    --soft-time-limit=150 --time-limit=180                      │
│    -Q steering                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task Breakdown

### Phase 1: Backend Infrastructure

#### Task 1.1: Create Steering Celery Task Module
**File:** `backend/src/workers/steering_tasks.py`

```python
"""
Celery tasks for steering operations.

These tasks run in a dedicated GPU worker process, providing:
- Process isolation (crashes don't affect API)
- Proper timeout handling via SIGKILL
- Worker recycling to prevent memory leaks
"""

import asyncio
import logging
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from typing import Dict, Any

from ..core.celery_app import celery
from ..services.steering_service import get_steering_service
from ..workers.websocket_emitter import emit_steering_progress

logger = logging.getLogger(__name__)


@celery.task(
    bind=True,
    name="steering.compare",
    queue="steering",
    soft_time_limit=150,  # SIGTERM after 150s
    time_limit=180,       # SIGKILL after 180s (guaranteed termination)
    max_retries=0,        # No retries for GPU tasks
    acks_late=True,       # Acknowledge after completion
    reject_on_worker_lost=True,  # Requeue if worker dies
    track_started=True,   # Track when task starts
)
def steering_compare_task(
    self,
    request_dict: Dict[str, Any],
    sae_id: str,
    model_id: str,
    sae_path: str,
    model_path: str = None,
    sae_layer: int = None,
    sae_d_model: int = None,
    sae_n_features: int = None,
    sae_architecture: str = None,
) -> Dict[str, Any]:
    """
    Execute steering comparison in isolated worker process.

    This task:
    1. Loads model and SAE to GPU
    2. Generates unsteered baseline
    3. Generates steered outputs for each feature/strength
    4. Emits progress via WebSocket
    5. Returns result dict

    Timeout behavior:
    - At 150s: SIGTERM sent, SoftTimeLimitExceeded raised
    - At 180s: SIGKILL sent, process terminated, GPU memory released
    """
    task_id = self.request.id
    logger.info(f"[Steering Task {task_id}] Starting steering comparison")

    try:
        # Emit initial progress
        emit_steering_progress(task_id, 0, "Initializing...")

        service = get_steering_service()

        # Run the synchronous steering operation
        # (We'll create a sync wrapper for the service)
        result = service.generate_comparison_sync(
            request_dict=request_dict,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
            sae_layer=sae_layer,
            sae_d_model=sae_d_model,
            sae_n_features=sae_n_features,
            sae_architecture=sae_architecture,
            progress_callback=lambda pct, msg: emit_steering_progress(task_id, pct, msg),
        )

        # Emit completion
        emit_steering_progress(task_id, 100, "Complete", result=result)

        logger.info(f"[Steering Task {task_id}] Completed successfully")
        return result

    except SoftTimeLimitExceeded:
        logger.warning(f"[Steering Task {task_id}] Soft time limit exceeded, cleaning up...")
        emit_steering_progress(task_id, -1, "Timeout - cleaning up", error="Task exceeded time limit")

        # Attempt graceful cleanup
        try:
            service = get_steering_service()
            service.cleanup_gpu()
        except Exception as e:
            logger.error(f"[Steering Task {task_id}] Cleanup failed: {e}")

        raise  # Re-raise to mark task as failed

    except Exception as e:
        logger.exception(f"[Steering Task {task_id}] Failed: {e}")
        emit_steering_progress(task_id, -1, f"Failed: {str(e)}", error=str(e))
        raise


@celery.task(
    bind=True,
    name="steering.sweep",
    queue="steering",
    soft_time_limit=300,  # Sweeps take longer
    time_limit=360,
    max_retries=0,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def steering_sweep_task(
    self,
    request_dict: Dict[str, Any],
    sae_id: str,
    model_id: str,
    sae_path: str,
    model_path: str = None,
    sae_layer: int = None,
    sae_d_model: int = None,
    sae_n_features: int = None,
    sae_architecture: str = None,
) -> Dict[str, Any]:
    """Execute strength sweep in isolated worker process."""
    task_id = self.request.id
    logger.info(f"[Sweep Task {task_id}] Starting strength sweep")

    try:
        emit_steering_progress(task_id, 0, "Initializing sweep...")

        service = get_steering_service()

        result = service.generate_strength_sweep_sync(
            request_dict=request_dict,
            sae_path=sae_path,
            model_id=model_id,
            model_path=model_path,
            sae_layer=sae_layer,
            sae_d_model=sae_d_model,
            sae_n_features=sae_n_features,
            sae_architecture=sae_architecture,
            progress_callback=lambda pct, msg: emit_steering_progress(task_id, pct, msg),
        )

        emit_steering_progress(task_id, 100, "Complete", result=result)

        logger.info(f"[Sweep Task {task_id}] Completed successfully")
        return result

    except SoftTimeLimitExceeded:
        logger.warning(f"[Sweep Task {task_id}] Soft time limit exceeded")
        emit_steering_progress(task_id, -1, "Timeout", error="Task exceeded time limit")
        try:
            get_steering_service().cleanup_gpu()
        except:
            pass
        raise

    except Exception as e:
        logger.exception(f"[Sweep Task {task_id}] Failed: {e}")
        emit_steering_progress(task_id, -1, f"Failed: {str(e)}", error=str(e))
        raise
```

**Subtasks:**
- [ ] Create `backend/src/workers/steering_tasks.py`
- [ ] Add task registration to `celery_app.py`
- [ ] Add `steering` queue to routing configuration

---

#### Task 1.2: Add WebSocket Progress Emission for Steering
**File:** `backend/src/workers/websocket_emitter.py` (extend existing)

```python
def emit_steering_progress(
    task_id: str,
    percent: int,
    message: str,
    current_feature: int = None,
    current_strength: float = None,
    result: dict = None,
    error: str = None,
) -> bool:
    """
    Emit steering progress via WebSocket.

    Channel: steering/{task_id}
    Event: steering:progress (in progress) or steering:completed/steering:failed
    """
    channel = f"steering/{task_id}"

    if percent < 0:
        event = "steering:failed"
    elif percent >= 100:
        event = "steering:completed"
    else:
        event = "steering:progress"

    data = {
        "task_id": task_id,
        "percent": percent,
        "message": message,
        "current_feature": current_feature,
        "current_strength": current_strength,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if result:
        data["result"] = result
    if error:
        data["error"] = error

    return _emit_to_channel(channel, event, data)
```

**Subtasks:**
- [ ] Add `emit_steering_progress()` to websocket_emitter.py
- [ ] Add steering WebSocket channel documentation

---

#### Task 1.3: Create Synchronous Steering Service Wrapper
**File:** `backend/src/services/steering_service.py` (extend existing)

The current service methods are async. We need synchronous versions for Celery tasks.

```python
def generate_comparison_sync(
    self,
    request_dict: Dict[str, Any],
    sae_path: str,
    model_id: str,
    model_path: Optional[str] = None,
    sae_layer: Optional[int] = None,
    sae_d_model: Optional[int] = None,
    sae_n_features: Optional[int] = None,
    sae_architecture: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_comparison.

    Used by Celery tasks where async is not needed.
    Progress callback allows emitting WebSocket updates.
    """
    # Convert request_dict back to Pydantic model
    from ..schemas.steering import SteeringComparisonRequest
    request = SteeringComparisonRequest(**request_dict)

    # Run the async method synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            self.generate_comparison(
                request=request,
                sae_path=Path(sae_path),
                model_id=model_id,
                model_path=model_path,
                sae_layer=sae_layer,
                sae_d_model=sae_d_model,
                sae_n_features=sae_n_features,
                sae_architecture=sae_architecture,
                progress_callback=progress_callback,
            )
        )
        return result.dict()
    finally:
        loop.close()
```

**Subtasks:**
- [ ] Add `generate_comparison_sync()` method
- [ ] Add `generate_strength_sweep_sync()` method
- [ ] Add `progress_callback` parameter to async methods
- [ ] Emit progress at key points (model load, each generation, completion)

---

#### Task 1.4: Update Async Methods with Progress Callbacks
**File:** `backend/src/services/steering_service.py`

Modify `generate_comparison()` and related methods to accept and call progress callbacks:

```python
async def generate_comparison(
    self,
    request: SteeringComparisonRequest,
    sae_path: Path,
    model_id: str,
    # ... existing params ...
    progress_callback: Optional[Callable[[int, str], None]] = None,  # NEW
) -> SteeringComparisonResponse:
    """..."""

    def emit_progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, message)

    emit_progress(5, "Loading SAE...")
    sae = await self.load_sae(...)

    emit_progress(15, "Loading model...")
    model, tokenizer = await self.load_model(...)

    emit_progress(25, "Generating unsteered baseline...")
    # ... generate unsteered ...

    # In the generation loop:
    total_generations = len(unique_features) * (1 + len(additional_strengths))
    for i, (feature, strength) in enumerate(generation_items):
        percent = 25 + int((i / total_generations) * 70)  # 25% to 95%
        emit_progress(percent, f"Generating feature {feature.feature_idx} @ {strength}")
        # ... generate ...

    emit_progress(95, "Computing metrics...")
    # ... compute metrics ...

    emit_progress(100, "Complete")
    return response
```

**Subtasks:**
- [ ] Add progress_callback to `generate_comparison()`
- [ ] Add progress_callback to `_generate_multi_strength_outputs()`
- [ ] Add progress_callback to `generate_strength_sweep()`
- [ ] Calculate progress percentages correctly for multi-feature/multi-strength

---

### Phase 2: API Layer

#### Task 2.1: Create Async Steering Endpoints
**File:** `backend/src/api/v1/endpoints/steering.py` (modify existing)

```python
# NEW: Async endpoint - submits to Celery
@router.post("/compare", response_model=SteeringTaskResponse)
async def submit_steering_comparison(
    request: SteeringComparisonRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a steering comparison task.

    Returns immediately with a task_id. Use WebSocket channel
    steering/{task_id} for progress updates, or poll
    GET /steering/result/{task_id} for completion.
    """
    # Rate limiting (existing)
    client_id = get_client_id(http_request)
    if not _rate_limiter.is_allowed(client_id):
        raise HTTPException(429, "Rate limit exceeded")

    # Circuit breaker (existing)
    circuit_breaker = get_circuit_breaker()
    can_execute, reason = await circuit_breaker.can_execute()
    if not can_execute:
        raise HTTPException(503, f"Steering unavailable: {reason}")

    # Validate SAE and model (existing validation logic)
    sae = await SAEManagerService.get_sae(db, request.sae_id)
    if not sae:
        raise HTTPException(404, f"SAE not found: {request.sae_id}")
    # ... more validation ...

    # Submit task to Celery
    from ..workers.steering_tasks import steering_compare_task

    task = steering_compare_task.delay(
        request_dict=request.dict(),
        sae_id=request.sae_id,
        model_id=model_id,
        sae_path=str(sae_path),
        model_path=model_path,
        sae_layer=sae.layer,
        sae_d_model=sae.d_model,
        sae_n_features=sae.n_features,
        sae_architecture=sae.architecture,
    )

    return SteeringTaskResponse(
        task_id=task.id,
        status="pending",
        message="Task submitted. Subscribe to WebSocket channel steering/{task_id} for progress.",
    )


@router.get("/result/{task_id}", response_model=SteeringResultResponse)
async def get_steering_result(task_id: str):
    """
    Get the result of a steering task.

    Returns task status and result if completed.
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id)

    response = SteeringResultResponse(
        task_id=task_id,
        status=result.status,
    )

    if result.ready():
        if result.successful():
            response.result = result.get()
            response.status = "completed"
        else:
            response.error = str(result.result)
            response.status = "failed"
    elif result.status == "STARTED":
        response.status = "running"
    elif result.status == "PENDING":
        response.status = "pending"

    return response


@router.delete("/task/{task_id}")
async def cancel_steering_task(task_id: str):
    """
    Cancel a running steering task.

    Sends SIGTERM to the worker. If task doesn't stop within
    30 seconds, Celery will send SIGKILL.
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id)
    result.revoke(terminate=True, signal='SIGTERM')

    return {"message": f"Task {task_id} cancellation requested"}


# LEGACY: Mark for deprecation
@router.post("/compare/sync", response_model=SteeringComparisonResponse, deprecated=True)
async def generate_steering_comparison_sync(
    request: SteeringComparisonRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    [DEPRECATED] Synchronous steering comparison.

    This endpoint runs steering in the API process, which can cause
    zombie processes if generation hangs. Use POST /compare instead.

    Will be removed in v2.0.
    """
    # ... existing implementation ...
```

**Subtasks:**
- [ ] Create `SteeringTaskResponse` schema
- [ ] Create `SteeringResultResponse` schema
- [ ] Implement `POST /compare` (async, returns task_id)
- [ ] Implement `GET /result/{task_id}` (poll status)
- [ ] Implement `DELETE /task/{task_id}` (cancel)
- [ ] Implement `POST /sweep` (async sweep)
- [ ] Mark existing sync endpoints as deprecated

---

#### Task 2.2: Add Steering Schemas for Async Flow
**File:** `backend/src/schemas/steering.py` (extend existing)

```python
class SteeringTaskResponse(BaseModel):
    """Response when submitting a steering task."""
    task_id: str
    status: str  # pending, running, completed, failed
    message: str
    websocket_channel: Optional[str] = None  # steering/{task_id}


class SteeringResultResponse(BaseModel):
    """Response when polling for task result."""
    task_id: str
    status: str
    result: Optional[SteeringComparisonResponse] = None
    error: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None
```

**Subtasks:**
- [ ] Add `SteeringTaskResponse` schema
- [ ] Add `SteeringResultResponse` schema
- [ ] Add `SteeringSweepTaskResponse` schema

---

### Phase 3: Frontend Integration

#### Task 3.1: Create useSteeringWebSocket Hook
**File:** `frontend/src/hooks/useSteeringWebSocket.ts`

```typescript
import { useEffect, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { WS_URL, WS_PATH } from '@/config/api';

interface SteeringProgress {
  task_id: string;
  percent: number;
  message: string;
  current_feature?: number;
  current_strength?: number;
  result?: any;
  error?: string;
}

export function useSteeringWebSocket(
  taskId: string | null,
  onProgress: (progress: SteeringProgress) => void,
  onCompleted: (result: any) => void,
  onFailed: (error: string) => void,
) {
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!taskId) return;

    const socket = io(WS_URL, {
      path: WS_PATH,
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;
    const channel = `steering/${taskId}`;

    socket.on('connect', () => {
      console.log(`[Steering WS] Connected, joining ${channel}`);
      socket.emit('join', { room: channel });
    });

    socket.on('steering:progress', (data: SteeringProgress) => {
      console.log('[Steering WS] Progress:', data);
      onProgress(data);
    });

    socket.on('steering:completed', (data: SteeringProgress) => {
      console.log('[Steering WS] Completed:', data);
      onCompleted(data.result);
    });

    socket.on('steering:failed', (data: SteeringProgress) => {
      console.log('[Steering WS] Failed:', data);
      onFailed(data.error || 'Unknown error');
    });

    return () => {
      socket.emit('leave', { room: channel });
      socket.disconnect();
    };
  }, [taskId, onProgress, onCompleted, onFailed]);

  return socketRef.current;
}
```

**Subtasks:**
- [ ] Create `useSteeringWebSocket.ts` hook
- [ ] Handle connection/disconnection
- [ ] Handle progress, completed, failed events

---

#### Task 3.2: Update steeringStore for Async Flow
**File:** `frontend/src/stores/steeringStore.ts` (modify existing)

```typescript
interface SteeringState {
  // ... existing state ...

  // Async task state
  currentTaskId: string | null;
  taskStatus: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
  taskProgress: number;
  taskMessage: string;

  // Actions
  submitSteeringTask: (request: SteeringComparisonRequest) => Promise<string>;
  pollTaskResult: (taskId: string) => Promise<void>;
  cancelTask: (taskId: string) => Promise<void>;
  updateProgress: (percent: number, message: string) => void;
  setTaskCompleted: (result: SteeringComparisonResponse) => void;
  setTaskFailed: (error: string) => void;
}

export const useSteeringStore = create<SteeringState>((set, get) => ({
  // ... existing state ...

  currentTaskId: null,
  taskStatus: 'idle',
  taskProgress: 0,
  taskMessage: '',

  submitSteeringTask: async (request) => {
    set({ taskStatus: 'pending', taskProgress: 0, taskMessage: 'Submitting...' });

    try {
      const response = await axios.post('/api/v1/steering/compare', request);
      const { task_id } = response.data;

      set({ currentTaskId: task_id, taskStatus: 'running' });
      return task_id;
    } catch (error) {
      set({ taskStatus: 'failed', taskMessage: error.message });
      throw error;
    }
  },

  updateProgress: (percent, message) => {
    set({ taskProgress: percent, taskMessage: message });
  },

  setTaskCompleted: (result) => {
    set({
      taskStatus: 'completed',
      taskProgress: 100,
      taskMessage: 'Complete',
      comparisonResult: result,  // existing result field
      currentTaskId: null,
    });
  },

  setTaskFailed: (error) => {
    set({
      taskStatus: 'failed',
      taskMessage: error,
      currentTaskId: null,
    });
  },

  cancelTask: async (taskId) => {
    try {
      await axios.delete(`/api/v1/steering/task/${taskId}`);
      set({ taskStatus: 'idle', currentTaskId: null });
    } catch (error) {
      console.error('Failed to cancel task:', error);
    }
  },
}));
```

**Subtasks:**
- [ ] Add async task state fields
- [ ] Add `submitSteeringTask()` action
- [ ] Add `updateProgress()` action
- [ ] Add `setTaskCompleted()` / `setTaskFailed()` actions
- [ ] Add `cancelTask()` action
- [ ] Integrate with existing result handling

---

#### Task 3.3: Update SteeringPanel UI
**File:** `frontend/src/components/panels/SteeringPanel.tsx` (modify existing)

```typescript
// Add progress indicator component
const SteeringProgressIndicator: React.FC<{
  progress: number;
  message: string;
  onCancel: () => void;
}> = ({ progress, message, onCancel }) => (
  <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
    <div className="flex items-center justify-between mb-2">
      <span className="text-sm text-slate-300">{message}</span>
      <button
        onClick={onCancel}
        className="text-xs text-red-400 hover:text-red-300"
      >
        Cancel
      </button>
    </div>
    <div className="w-full bg-slate-700 rounded-full h-2">
      <div
        className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
        style={{ width: `${progress}%` }}
      />
    </div>
    <span className="text-xs text-slate-500 mt-1">{progress}%</span>
  </div>
);

// In SteeringPanel component:
const SteeringPanel: React.FC = () => {
  const {
    currentTaskId,
    taskStatus,
    taskProgress,
    taskMessage,
    submitSteeringTask,
    updateProgress,
    setTaskCompleted,
    setTaskFailed,
    cancelTask,
  } = useSteeringStore();

  // Connect WebSocket when task is running
  useSteeringWebSocket(
    taskStatus === 'running' ? currentTaskId : null,
    (progress) => updateProgress(progress.percent, progress.message),
    (result) => setTaskCompleted(result),
    (error) => setTaskFailed(error),
  );

  const handleGenerate = async () => {
    try {
      await submitSteeringTask(buildRequest());
      // WebSocket will handle progress updates
    } catch (error) {
      toast.error(`Failed to submit: ${error.message}`);
    }
  };

  return (
    <div>
      {/* ... existing UI ... */}

      {taskStatus === 'running' && (
        <SteeringProgressIndicator
          progress={taskProgress}
          message={taskMessage}
          onCancel={() => cancelTask(currentTaskId!)}
        />
      )}

      <button
        onClick={handleGenerate}
        disabled={taskStatus === 'running'}
        className={taskStatus === 'running' ? 'opacity-50' : ''}
      >
        {taskStatus === 'running' ? 'Generating...' : 'Generate'}
      </button>
    </div>
  );
};
```

**Subtasks:**
- [ ] Create `SteeringProgressIndicator` component
- [ ] Integrate `useSteeringWebSocket` hook
- [ ] Update generate button state
- [ ] Add cancel functionality
- [ ] Show progress during generation

---

### Phase 4: Infrastructure

#### Task 4.1: Configure Steering Queue and Worker
**File:** `backend/src/core/celery_app.py` (modify existing)

```python
# Add steering queue to task routes
celery.conf.task_routes = {
    # ... existing routes ...
    'steering.*': {'queue': 'steering'},
}

# Add steering to autodiscover
celery.autodiscover_tasks([
    'src.workers.dataset_tasks',
    'src.workers.training_tasks',
    'src.workers.extraction_tasks',
    'src.workers.sae_tasks',
    'src.workers.system_monitor_tasks',
    'src.workers.steering_tasks',  # NEW
])
```

**Subtasks:**
- [ ] Add `steering` queue to task_routes
- [ ] Add `steering_tasks` to autodiscover
- [ ] Update celery app configuration

---

#### Task 4.2: Add GPU Worker Script
**File:** `celery.sh` (modify existing)

```bash
# Add steering worker command
start_steering_worker() {
    echo "Starting Steering GPU Worker..."
    celery -A src.core.celery_app worker \
        -Q steering \
        --pool=solo \
        --concurrency=1 \
        --max-tasks-per-child=50 \
        --soft-time-limit=150 \
        --time-limit=180 \
        --loglevel=info \
        --hostname=steering@%h \
        --pidfile=/tmp/mistudio-celery-steering.pid \
        &
    echo "Steering worker started"
}
```

**Subtasks:**
- [ ] Add `start_steering_worker()` function to celery.sh
- [ ] Add steering worker to start-mistudio.sh
- [ ] Add steering worker to stop-mistudio.sh
- [ ] Configure worker restart on failure

---

### Phase 5: Legacy Code Cleanup

#### Task 5.1: Mark Legacy Code for Deprecation

**Files to mark as deprecated:**
1. `backend/src/api/v1/endpoints/steering.py`:
   - `generate_steering_comparison()` → rename to `generate_steering_comparison_sync()`, add `deprecated=True`
   - `generate_strength_sweep()` → rename to `generate_strength_sweep_sync()`, add `deprecated=True`

2. `backend/src/services/steering_service.py`:
   - Remove `GenerationWatchdog` class after Celery migration (Celery handles timeouts)
   - Keep `_emergency_gpu_cleanup()` and signal handlers (still useful for graceful shutdown)

3. `backend/src/services/steering_resilience.py`:
   - `ProcessIsolationManager` → Can be removed (Celery provides this)
   - Keep `CircuitBreaker` and `ConcurrencyLimiter` (still useful for API-level protection)

**Subtasks:**
- [ ] Mark sync endpoints as deprecated with warning in response
- [ ] Add deprecation notice to API docs
- [ ] Plan removal date (e.g., v2.0)
- [ ] Remove GenerationWatchdog after Celery confirmed working
- [ ] Remove ProcessIsolationManager after Celery confirmed working

---

## Testing Plan

### Unit Tests
- [ ] Test steering_compare_task with mock service
- [ ] Test steering_sweep_task with mock service
- [ ] Test progress callback emission
- [ ] Test timeout handling (mock SoftTimeLimitExceeded)

### Integration Tests
- [ ] Test full flow: POST → WebSocket progress → GET result
- [ ] Test cancellation: POST → DELETE → verify worker stopped
- [ ] Test timeout: Long-running task → verify SIGKILL → verify worker restart
- [ ] Test circuit breaker integration

### Manual Tests
- [ ] Run multi-feature steering, verify progress updates
- [ ] Run multi-strength steering, verify progress updates
- [ ] Intentionally cause timeout, verify no zombie process
- [ ] Verify GPU memory released after task completion/failure

---

## Rollout Plan

1. **Phase 1**: Deploy Celery tasks alongside existing endpoints (no breaking changes)
2. **Phase 2**: Update frontend to use async endpoints by default
3. **Phase 3**: Monitor for 1-2 weeks, verify no issues
4. **Phase 4**: Mark sync endpoints as deprecated
5. **Phase 5**: Remove sync endpoints in next major version

---

## Configuration

Add to `backend/src/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Steering task settings
    steering_soft_time_limit: int = 150  # seconds
    steering_hard_time_limit: int = 180  # seconds
    steering_max_tasks_per_child: int = 50
```

---

## Future Containerization Notes

For Kubernetes deployment:

```yaml
# Separate deployment for steering GPU worker
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistudio-steering-worker
spec:
  replicas: 1  # Scale based on GPU availability
  template:
    spec:
      containers:
      - name: worker
        resources:
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          exec:
            command: ["celery", "inspect", "ping"]
          periodSeconds: 30
          failureThreshold: 3
          # K8s will restart pod if unhealthy
          # GPU memory automatically released on pod termination
```

This architecture enables:
- Independent scaling of API and GPU workers
- GPU resource isolation per worker pod
- Automatic recovery from crashes via K8s restart policy
- No zombie processes (container termination releases all resources)
