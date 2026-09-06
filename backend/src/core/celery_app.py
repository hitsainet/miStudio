"""
Celery application configuration for background task processing.

This module initializes Celery with Redis broker for distributed task queue
processing, including dataset downloads, tokenization, and training jobs.

⚠️ IMPORTANT: Worker Startup Configuration
===========================================
Workers MUST be started with explicit queue configuration using the -Q flag!

❌ WRONG (will only listen to default "celery" queue):
    celery -A src.core.celery_app worker --loglevel=info

✅ CORRECT (listens to all required queues):
    celery -A src.core.celery_app worker -Q high_priority,datasets,processing,training,extraction,low_priority -c 8 --loglevel=info

OR use the startup script:
    ./backend/start-celery-worker.sh

See backend/CELERY_WORKERS.md for full documentation.
"""

from celery import Celery
from celery.signals import task_failure, task_success, worker_ready

from .config import settings

# Apply transformers compatibility patches BEFORE any task imports
# This prevents import errors during autodiscovery
from ..ml.transformers_compat import patch_transformers_compatibility
patch_transformers_compatibility()

# Initialize Celery app
celery_app = Celery(
    "mistudio",
    broker=str(settings.celery_broker_url),
    backend=str(settings.celery_result_backend),
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # ==========================================
    # BROKER CONNECTION RELIABILITY SETTINGS
    # ==========================================
    # These settings ensure Celery automatically recovers from
    # Redis disconnections without manual intervention

    # Retry connecting to broker on startup (waits for Redis)
    broker_connection_retry_on_startup=True,

    # Keep retrying broker connection indefinitely
    broker_connection_retry=True,
    broker_connection_max_retries=None,  # Infinite retries

    # Connection pool settings for stability
    broker_pool_limit=10,  # Limit connection pool size
    broker_heartbeat=30,   # Send heartbeat every 30 seconds
    broker_heartbeat_checkrate=2.0,  # Check heartbeat twice per interval

    # Transport options for Redis reliability
    broker_transport_options={
        "visibility_timeout": 43200,  # 12 hours - match task_time_limit
        "socket_timeout": 30,
        "socket_connect_timeout": 30,
        "retry_on_timeout": True,
        "health_check_interval": 30,  # Check connection health every 30s
    },

    # Result backend connection settings
    redis_socket_timeout=30,
    redis_socket_connect_timeout=30,
    redis_retry_on_timeout=True,

    # Task routing - Multi-queue architecture for optimal resource allocation
    # NOTE: Priority values removed since task_queue_max_priority is disabled
    task_routes={
        # High priority: Quick operations, metadata updates
        "src.workers.quick_tasks.*": {
            "queue": "high_priority",
        },

        # Dataset operations: I/O bound, medium concurrency
        "src.workers.dataset_tasks.download_dataset_task": {
            "queue": "datasets",
        },
        "src.workers.dataset_tasks.tokenize_dataset_task": {
            "queue": "datasets",  # Changed from "processing" to "datasets" for consistency
        },
        # The two dataset tasks nothing matched (MIS-E2E-093/-097).
        "src.workers.dataset_tasks.cancel_dataset_download": {
            "queue": "high_priority",       # must preempt the download it stops
        },
        "src.workers.dataset_tasks.delete_dataset_files": {
            "queue": "low_priority",
        },

        # Model operations: GPU bound, high priority.
        #
        # THE PREFIX HERE WAS WRONG AND SILENTLY MATCHED NOTHING
        # (MIS-E2E-093/-097). These tasks register as `workers.model_tasks.*`,
        # not `src.workers.model_tasks.*`, so the entry below never applied and
        # every model task — including `extract_activations`, a GPU job — ran on
        # the default queue. This is the recorded lesson biting a second time:
        # `task_routes` globs match the TASK NAME, and the name is whatever the
        # decorator registered, not the import path.
        "workers.model_tasks.download_and_load_model": {
            "queue": "high_priority",
        },
        "workers.model_tasks.extract_activations": {
            "queue": "extraction",          # GPU: the model is resident
        },
        "workers.model_tasks.cancel_download": {
            "queue": "high_priority",       # must preempt the download it stops
        },
        "workers.model_tasks.delete_model_files": {
            "queue": "low_priority",        # I/O cleanup
        },
        "workers.model_tasks.update_model_progress": {
            "queue": "high_priority",       # a metadata write on a hot path
        },

        # Training operations: GPU bound, low concurrency
        "src.workers.training_tasks.*": {
            "queue": "training",
        },
        # THE SHORT NAMES THE MODULE GLOB ABOVE CANNOT MATCH (MIS-E2E-093/-097).
        #
        # `task_routes` patterns match the TASK NAME, and these three tasks are
        # registered under bare names, so `src.workers.training_tasks.*` never
        # applied to them and they fell to the DEFAULT queue — two GPU training
        # jobs among them, competing with every quick task on the same worker.
        #
        # Found by `test_every_registered_task_routes_to_its_intended_queue`,
        # which is driven off the live registry: 16 tasks carry short names and
        # 13 already had explicit entries here. These three did not, and nothing
        # would have said so.
        "train_sae": {
            "queue": "training",
        },
        "resume_training": {
            "queue": "training",
        },
        "delete_extraction": {
            "queue": "extraction",
        },

        # Extraction operations: GPU bound, medium concurrency
        "src.workers.extraction_tasks.*": {
            "queue": "extraction",
        },

        # Circuit capture + attribution (Feature 016): GPU profile — share
        # the extraction queue/worker.
        "src.workers.circuit_capture_tasks.capture_circuit_activations": {
            "queue": "extraction",
        },
        "src.workers.circuit_capture_tasks.run_circuit_attribution": {
            "queue": "extraction",
        },
        # Circuit validation (Feature 017): GPU — share the extraction queue.
        "src.workers.circuit_validation_tasks.*": {
            "queue": "extraction",
        },
        # Circuit calibration (Feature 20): GPU (loads the model to generate) —
        # same extraction queue + single-GPU circuit guard as validation.
        "src.workers.circuit_calibration_tasks.*": {
            "queue": "extraction",
        },
        # J-lens fitting (feature 022): GPU — the whole model is resident and
        # every layer takes a linearised pass. Same extraction queue as the
        # other GPU jobs. The name here is FULLY QUALIFIED because task_routes
        # globs match the TASK NAME, not the module path — a short name lands
        # on the default queue silently.
        "src.workers.jlens_fit_tasks.*": {
            "queue": "extraction",
        },
        # J-lens readout (feature 022 Phase 4.5): the whole model must be
        # resident for the forward pass, so this is model-bound like the rest
        # of the extraction queue. It lives in the WORKER rather than the API
        # because a synchronous handler 502'd at the 60s ingress timeout, and
        # because only the worker can hold the loaded model across requests.
        "src.workers.jlens_readout_tasks.*": {
            "queue": "extraction",
        },
        # J-lens probe: same model-bound forward pass as the readout, so the
        # same queue — sharing it also shares the worker's single-entry model
        # cache, which is the whole reason the readout moved here.
        "src.workers.jlens_probe_tasks.*": {
            "queue": "extraction",
        },
        # Band-report computation: model-bound, and the ONLY thing that can
        # make bands appear anywhere in the product (BR-002).
        "src.workers.jlens_band_tasks.*": {
            "queue": "extraction",
        },
        # J-space interventions: a forward pass plus its matched control, so
        # model-bound like the rest.
        # ENUMERATED, LIKE ITS FOUR SIBLINGS. There is no `jlens_*` glob here —
        # a new module needs an entry in THIS list and another in
        # `autodiscover_tasks` below, and the right task name is necessary but
        # not sufficient for either.
        "src.workers.jlens_acquire_tasks.*": {
            "queue": "extraction",
        },
        "src.workers.jlens_intervention_tasks.*": {
            "queue": "extraction",
        },
        # Steered-transcript recorder: GPU (loads the model to generate) — same
        # extraction queue + single-GPU guard as calibration.
        "src.workers.circuit_record_tasks.*": {
            "queue": "extraction",
        },
        # Circuit discovery is CPU-only (statistical mining) — route to the
        # CPU processing queue so it never head-of-line-blocks GPU work
        # behind a multi-minute mine (R1 QA-P1).
        "src.workers.circuit_capture_tasks.run_circuit_discovery": {
            "queue": "processing",
        },

        # Labeling operations: LLM bound, medium priority
        "src.workers.labeling_tasks.*": {
            "queue": "processing",
        },
        "src.workers.enhanced_labeling_tasks.*": {
            "queue": "processing",
        },
        # Explicit routes for tasks with custom names (module-path pattern won't match)
        "label_features": {
            "queue": "processing",
        },
        "enhanced_label_feature": {
            "queue": "processing",
        },

        # NLP analysis operations: CPU/network bound (HTTP calls to LLM server)
        # Routed to low_priority so GPU-bound extractions aren't blocked
        "src.workers.nlp_analysis_tasks.*": {
            "queue": "low_priority",
        },

        # Cross-feature grouping precompute (Feature 010): CPU-only
        "src.workers.feature_grouping_tasks.*": {
            "queue": "low_priority",
        },

        # Finalize a stopped training from a checkpoint (Feature 021): CPU-only.
        # Deliberately NOT on the "training" queue — rebuilding SAE modules and
        # writing community_format never touches the GPU, so a finalize must be
        # able to run while another training owns the device.
        "src.workers.training_finalize_tasks.*": {
            "queue": "low_priority",
        },

        # Checkpoint retention pruning (Feature 021): CPU + filesystem only.
        "src.workers.prune_checkpoints.*": {
            "queue": "low_priority",
        },

        # Maintenance operations: Background tasks
        "src.workers.maintenance_tasks.*": {
            "queue": "low_priority",
        },

        # Cleanup operations: Periodic maintenance tasks
        # ---------------------------------------------------------------
        # Janitors and periodic tasks, routed by their EXACT task name.
        #
        # These are registered with SHORT names ("cleanup_stuck_extractions"),
        # and task_routes globs match the TASK NAME — not the module path. So
        # "src.workers.cleanup_stuck_extractions.*" never matched, and every one
        # of these resolved to the DEFAULT queue ("datasets") when called
        # directly. The beat entries pass an explicit options.queue, which is the
        # only reason the scheduled runs land correctly; any .delay() call put a
        # CPU janitor on the GPU worker's queue.
        #
        # Verified by resolving celery_app.amqp.router.route({}, name), not by
        # reading the config back — reading it back is what hid this.
        # ---------------------------------------------------------------
        "cleanup_stuck_nlp": {"queue": "low_priority"},
        "cleanup_stuck_extractions": {"queue": "low_priority"},
        "cleanup_stuck_tokenizations": {"queue": "low_priority"},
        "cleanup_stuck_trainings": {"queue": "low_priority"},
        "cleanup_stuck_activations": {"queue": "low_priority"},
        "cleanup_stuck_circuit_runs": {"queue": "low_priority"},
        "cleanup_stuck_labeling": {"queue": "low_priority"},
        "cleanup_stuck_enhanced_labeling": {"queue": "low_priority"},
        "cleanup_task_queue": {"queue": "low_priority"},
        # SHORT NAME, explicit queue — the trap documented above. A glob on the
        # module path would never match this task.
        "cleanup_orphaned_tasks": {"queue": "low_priority"},
        "gpu_watchdog": {"queue": "low_priority"},
        "steering_worker_reconcile": {"queue": "low_priority"},

        # Kept for tasks invoked by their fully-qualified name.
        "src.workers.cleanup_stuck_nlp.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_stuck_extractions.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_stuck_tokenizations.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_stuck_trainings.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_stuck_activations.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_stuck_labeling.*": {"queue": "low_priority"},
        "src.workers.cleanup_stuck_enhanced_labeling.*": {
            "queue": "low_priority",
        },
        "src.workers.cleanup_task_queue.*": {
            "queue": "low_priority",
        },

        # SAE operations: HuggingFace downloads/uploads
        "src.workers.sae_tasks.*": {
            "queue": "sae",
        },
        "sae.download": {
            "queue": "sae",
        },
        "sae.upload": {
            "queue": "sae",
        },

        # Neuronpedia export operations: GPU bound, low concurrency
        "src.workers.neuronpedia_tasks.*": {
            "queue": "processing",
        },
        "neuronpedia.execute_export": {
            "queue": "processing",
        },
        "neuronpedia.compute_dashboard_data": {
            "queue": "processing",
        },

        # Neuronpedia push operations: Database bound, medium priority
        "src.workers.neuronpedia_push_tasks.*": {
            "queue": "processing",
        },
        "push_to_neuronpedia_local": {
            "queue": "processing",
        },

        # Steering operations: GPU bound, dedicated queue for isolation
        # Steering runs in a dedicated worker with --max-tasks-per-child for memory cleanup
        "src.workers.steering_tasks.*": {
            "queue": "steering",
        },
        # A GLOB, not two names. `steering.combined` — the multi-feature GPU
        # path — and `steering.cleanup` were both missing, so they ran on the
        # default queue while `compare` and `sweep` beside them did not
        # (MIS-E2E-093/-097).
        "steering.*": {
            "queue": "steering",
        },
        "steering.compare": {
            "queue": "steering",
        },
        "steering.sweep": {
            "queue": "steering",
        },
    },

    # Task priority queues (higher priority = processed first)
    # NOTE: Disabled priority suffixes to prevent Kombu from creating priority-based queue names
    # task_queue_max_priority=10,
    # task_default_priority=5,

    # Queue configuration
    task_default_queue="datasets",  # Default queue for unrouted tasks
    task_create_missing_queues=True,  # Auto-create queues if they don't exist

    # Task time limits (soft/hard)
    # Training tasks can take 5-10 hours for 100k steps, so set generous limits
    task_soft_time_limit=36000,  # 10 hour soft limit (training tasks need this)
    task_time_limit=43200,  # 12 hour hard limit (safety margin)

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mistudio",
    },

    # Task execution options
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Reject tasks if worker crashes
    worker_prefetch_multiplier=1,  # Disable prefetching for fair distribution

    # Beat scheduler settings (for periodic tasks)
    # NOTE: System metrics monitoring runs as an asyncio background task in
    # src/services/background_monitor.py (not Celery) so it can't be blocked
    # by long-running Celery tasks.
    # MIS-E2E-095: every entry carries `expires`. Without one, a tick that
    # queues behind a long GPU task is still delivered when the queue drains —
    # so an hour of blocked ticks all fire at once, against state that has
    # moved on. Each expires at 90% of its own period: past that a fresh tick
    # is already due, and the stale one has nothing to add.
    beat_schedule={
        # Cleanup stuck NLP analysis passes - runs every 10 minutes.
        #
        # nlp_status had no janitor: an NLP pass whose worker died left the row
        # reading "processing" forever, because every other cleanup_stuck_*
        # watches ExtractionJob.status, not nlp_status.
        #
        # The explicit queue matters. task_routes globs match the TASK NAME, and
        # this task's name is the short "cleanup_stuck_nlp" — it does NOT match
        # "src.workers.cleanup_stuck_nlp.*", so without this option it would
        # silently land on the default queue.
        "cleanup-stuck-nlp": {
            "task": "cleanup_stuck_nlp",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup stuck extraction jobs - runs every 10 minutes
        # Tokenization was the only long-running status with no janitor, so a
        # lost worker held PROCESSING forever with a frozen progress bar.
        "cleanup-stuck-tokenizations": {
            "task": "cleanup_stuck_tokenizations",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                "expires": 540.0,
            },
        },
        "cleanup-stuck-extractions": {
            "task": "cleanup_stuck_extractions",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup stuck circuit runs (Feature 016, R2 Q3): a died capture
        # leaves 'running' and would lock out every future capture via the
        # single-GPU guard. Runs every 10 minutes.
        "cleanup-stuck-circuit-runs": {
            "task": "cleanup_stuck_circuit_runs",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup stuck training jobs - runs every 10 minutes
        "cleanup-stuck-trainings": {
            "task": "cleanup_stuck_trainings",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup stuck activation extraction jobs - runs every 10 minutes
        "cleanup-stuck-activations": {
            "task": "cleanup_stuck_activations",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup stuck enhanced labeling jobs - runs every 5 minutes
        # Short threshold because enhanced labeling jobs are short-lived (seconds to minutes)
        "cleanup-stuck-enhanced-labeling": {
            "task": "cleanup_stuck_enhanced_labeling",
            "schedule": 300.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 270.0,
            },
        },
        # Cleanup stuck BULK labeling jobs - runs every 10 minutes.
        # 45-minute threshold, not the enhanced sibling's 10: bulk labeling
        # legitimately runs for hours over tens of thousands of features.
        # Without this, a job orphaned by a worker restart 409s every future
        # labeling run on its extraction until someone deletes it by hand.
        "cleanup-stuck-labeling": {
            "task": "cleanup_stuck_labeling",
            "schedule": 600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 540.0,
            },
        },
        # Cleanup old task_queue entries - runs hourly
        # Deletes completed entries >7 days old and stale queued/running ghosts
        "cleanup-task-queue": {
            "task": "cleanup_task_queue",
            "schedule": 3600.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 3240.0,
            },
        },
        # Close rows whose worker stopped reporting — every 5 minutes.
        #
        # A row is written when a task is QUEUED and moved by the task ITSELF,
        # so a worker killed by a pod roll writes nothing and the row keeps its
        # last progress forever. A J-lens fit sat at "running 21.5%" in Active
        # Operations for hours while the GPU was idle at 0%.
        #
        # FIVE MINUTES, not an hour: this is what a user is looking at while
        # wondering whether their job is alive, and the heartbeat threshold is
        # already ten minutes, so an hourly sweep would leave a ghost visible
        # for up to seventy.
        "cleanup-orphaned-tasks": {
            "task": "cleanup_orphaned_tasks",
            "schedule": 300.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 270.0,
            },
        },
        # Checkpoint retention pruning (Feature 021) - runs daily.
        # No-ops unless checkpoint_prune_enabled=true, and reports without
        # deleting while checkpoint_prune_dry_run=true (both defaults).
        "prune-checkpoints": {
            "task": "src.workers.prune_checkpoints.prune_checkpoints",
            "schedule": 86400.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 77760.0,
            },
        },
        # GPU Memory Watchdog - runs every minute to detect stuck processes
        # This is critical for preventing zombie processes from holding GPU memory
        "gpu-memory-watchdog": {
            "task": "gpu_watchdog",
            "schedule": 60.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 54.0,
            },
        },
        # Steering worker reconcile — respawns the (self-exiting) steering
        # worker when tasks are stranded on its queue with no consumer.
        "steering-worker-reconcile": {
            "task": "steering_worker_reconcile",
            "schedule": 30.0,
            "options": {
                "queue": "low_priority",
                # MIS-E2E-095: drop a tick that queued behind a long job.
                "expires": 27.0,
            },
        },
    },

    # Worker settings
    #
    # IMPORTANT: Workers MUST use --pool=solo for CUDA/GPU tasks.
    # This is set at runtime via command line (celery.sh, start-celery-worker.sh).
    # Why: Celery's default prefork pool uses fork(), which breaks CUDA because
    # CUDA maintains state in the parent process that cannot be forked safely.
    # Error without solo: "Cannot re-initialize CUDA in forked subprocess"
    #
    # max_tasks_per_child does NOTHING under --pool=solo. MIS-E2E-094.
    #
    # This setting recycles a prefork *child* process after N tasks. The solo
    # pool has no child — it runs each task in the worker's own process — so
    # Celery discards the value. The comment that used to sit here said this
    # setting recycled the solo worker and reclaimed its GPU memory; that
    # reclaim has never happened. `docker-entrypoint.sh` also passes
    # `--max-tasks-per-child`, equally inertly.
    #
    # VRAM is actually reclaimed by explicit `torch.cuda.empty_cache()` calls
    # in the GPU paths (55 sites across activation, steering, training, jlens
    # and the extraction services) and, for a process that dies holding memory,
    # by `workers/gpu_watchdog_task.py`. Those are the mechanisms to change if
    # reclaim is wrong — not this line.
    #
    # It is kept, at its original value, because a deployment that switches to
    # --pool=prefork for the CPU-only queue gets the recycling it implies. It
    # is no longer described as doing something it does not do.
    worker_max_tasks_per_child=100,
    worker_disable_rate_limits=False,

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task autodiscovery - automatically discover tasks in these modules
celery_app.autodiscover_tasks(
    [
        "src.workers.dataset_tasks",
        "src.workers.model_tasks",
        "src.workers.training_tasks",
        "src.workers.extraction_tasks",
        "src.workers.labeling_tasks",
        "src.workers.cleanup_stuck_extractions",
        "src.workers.cleanup_stuck_tokenizations",
        "src.workers.cleanup_stuck_nlp",
        "src.workers.cleanup_stuck_trainings",
        "src.workers.cleanup_stuck_activations",
        "src.workers.cleanup_orphaned_tasks",
        "src.workers.jlens_fit_tasks",
        "src.workers.jlens_readout_tasks",
        "src.workers.jlens_probe_tasks",
        "src.workers.jlens_band_tasks",
        "src.workers.jlens_acquire_tasks",
        "src.workers.jlens_intervention_tasks",
        "src.workers.cleanup_stuck_labeling",
        "src.workers.cleanup_stuck_enhanced_labeling",
        "src.workers.cleanup_task_queue",  # Old task_queue entry cleanup
        "src.workers.gpu_watchdog_task",  # GPU memory watchdog for detecting stuck processes
        "src.workers.steering_reconcile_task",  # Respawn steering worker for stranded queue
        "src.workers.sae_tasks",
        "src.workers.neuronpedia_tasks",
        "src.workers.neuronpedia_push_tasks",  # Push to local Neuronpedia
        "src.workers.nlp_analysis_tasks",
        "src.workers.feature_grouping_tasks",  # Cross-feature grouping precompute (Feature 010)
        "src.workers.steering_tasks",  # Steering operations in dedicated GPU worker
        "src.workers.enhanced_labeling_tasks",  # Enhanced per-feature two-pass labeling
        "src.workers.circuit_capture_tasks",  # Circuit capture/discovery/attribution (Feature 016)
        "src.workers.cleanup_stuck_circuit_runs",  # Reclaim stuck circuit runs (Feature 016)
        "src.workers.circuit_validation_tasks",  # Circuit edge validation (Feature 017)
        "src.workers.circuit_calibration_tasks",  # Circuit strength calibration (Feature 20)
        "src.workers.circuit_record_tasks",  # Steered-transcript recorder
        "src.workers.training_finalize_tasks",  # Finalize stopped training from checkpoint (Feature 021)
        "src.workers.prune_checkpoints",  # Checkpoint retention pruning (Feature 021)
    ],
    force=True,
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Signal handler called when Celery worker is ready.

    Args:
        sender: Worker instance
        **kwargs: Additional arguments
    """
    print(f"Celery worker ready: {sender.hostname}")


@task_success.connect
def on_task_success(sender=None, result=None, **kwargs):
    """
    Signal handler called when task completes successfully.

    Args:
        sender: Task instance
        result: Task result
        **kwargs: Additional arguments
    """
    if settings.is_development:
        print(f"Task success: {sender.name} - Result: {result}")


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """
    Signal handler called when task fails.

    Args:
        sender: Task instance
        task_id: Task ID
        exception: Exception that caused failure
        **kwargs: Additional arguments
    """
    print(f"Task failure: {sender.name} (ID: {task_id}) - Error: {exception}")


def create_task_signature(task_name: str, args: tuple = (), kwargs: dict = None, **options):
    """
    Create a task signature for delayed execution.

    Args:
        task_name: Full task name (e.g., 'app.workers.dataset_tasks.download_dataset_task')
        args: Positional arguments for task
        kwargs: Keyword arguments for task
        **options: Additional Celery options (countdown, eta, priority, etc.)

    Returns:
        celery.canvas.Signature: Task signature

    Usage:
        ```python
        from app.core.celery_app import create_task_signature

        # Create signature
        sig = create_task_signature(
            'app.workers.dataset_tasks.download_dataset_task',
            args=('ds_123', 'roneneldan/TinyStories'),
            priority=9
        )

        # Execute task
        result = sig.apply_async()
        ```
    """
    kwargs = kwargs or {}
    return celery_app.signature(task_name, args=args, kwargs=kwargs, **options)


def get_task_status(task_id: str) -> dict:
    """
    Get status of a Celery task.

    Args:
        task_id: Task ID returned from apply_async()

    Returns:
        dict: Task status information
            - state: PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
            - result: Task result (if completed)
            - traceback: Error traceback (if failed)
            - info: Additional task information

    Usage:
        ```python
        from app.core.celery_app import get_task_status

        status = get_task_status('task-uuid-123')
        if status['state'] == 'SUCCESS':
            print(f"Result: {status['result']}")
        elif status['state'] == 'FAILURE':
            print(f"Error: {status['traceback']}")
        ```
    """
    result = celery_app.AsyncResult(task_id)
    return {
        "state": result.state,
        "result": result.result if result.successful() else None,
        "traceback": result.traceback if result.failed() else None,
        "info": result.info,
    }


def revoke_task(task_id: str, terminate: bool = False) -> None:
    """
    Revoke (cancel) a Celery task.

    Args:
        task_id: Task ID to revoke
        terminate: If True, terminate task immediately (sends SIGTERM)
                  If False, task won't start if not already running

    Usage:
        ```python
        from app.core.celery_app import revoke_task

        # Stop task gracefully (if not started)
        revoke_task('task-uuid-123')

        # Force terminate running task
        revoke_task('task-uuid-123', terminate=True)
        ```

    Notes:
        - Terminated tasks cannot be recovered
        - Use terminate=True only when necessary
        - Task status will be set to REVOKED
    """
    celery_app.control.revoke(task_id, terminate=terminate)


def get_queue_lengths() -> dict:
    """
    Get the length of all Celery queues.

    Returns:
        dict: Queue names mapped to number of pending tasks

    Usage:
        ```python
        from app.core.celery_app import get_queue_lengths

        lengths = get_queue_lengths()
        # {'high_priority': 0, 'datasets': 2, 'processing': 1, ...}
        ```
    """
    from kombu import Connection

    queue_names = [
        "high_priority",
        "datasets",
        "processing",
        "training",
        "extraction",
        "sae",
        "steering",
        "low_priority",
    ]

    with Connection(str(settings.celery_broker_url)) as conn:
        lengths = {}
        for queue_name in queue_names:
            try:
                queue = conn.SimpleQueue(queue_name)
                lengths[queue_name] = queue.qsize()
                queue.close()
            except Exception as e:
                # Queue doesn't exist yet or error accessing it
                lengths[queue_name] = 0

        return lengths


def get_active_tasks() -> dict:
    """
    Get currently active (running) tasks across all workers.

    Returns:
        dict: Worker names mapped to list of active tasks

    Usage:
        ```python
        from app.core.celery_app import get_active_tasks

        active = get_active_tasks()
        # {
        #     'celery@worker1': [
        #         {'id': 'task-123', 'name': 'download_dataset_task', ...}
        #     ]
        # }
        ```
    """
    inspect = celery_app.control.inspect()
    active = inspect.active()
    return active or {}


def get_worker_stats() -> dict:
    """
    Get statistics for all connected workers.

    Returns:
        dict: Worker statistics including queue assignments and resource usage

    Usage:
        ```python
        from app.core.celery_app import get_worker_stats

        stats = get_worker_stats()
        ```
    """
    inspect = celery_app.control.inspect()

    stats = {}
    active_queues = inspect.active_queues()
    stats_data = inspect.stats()

    if active_queues:
        for worker, queues in active_queues.items():
            worker_stats = {
                "queues": [q["name"] for q in queues],
                "queue_details": queues,
            }

            # Add additional stats if available
            if stats_data and worker in stats_data:
                worker_stats["stats"] = stats_data[worker]

            stats[worker] = worker_stats

    return stats


# Export commonly used objects
__all__ = [
    "celery_app",
    "create_task_signature",
    "get_task_status",
    "revoke_task",
    "get_queue_lengths",
    "get_active_tasks",
    "get_worker_stats",
]
