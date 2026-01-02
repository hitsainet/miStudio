"""
Steering Resilience Infrastructure.

Provides circuit breaker, concurrency control, and process isolation
for the steering service to improve reliability and fault tolerance.

Components:
1. CircuitBreaker - Prevents cascading failures after repeated errors
2. ConcurrencyLimiter - Ensures only one steering request at a time
3. SteeringWorkerPool - Process isolation for GPU operations
"""

import asyncio
import enum
import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failures exceeded threshold, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 3  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    half_open_max_calls: int = 1  # Test requests in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    total_rejected: int
    time_until_retry: Optional[float]  # Seconds until half-open (if open)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for steering.

    Prevents cascading failures by temporarily blocking requests
    after repeated failures, giving the system time to recover.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Blocking all requests, waiting for recovery timeout
    - HALF_OPEN: Allowing limited test requests to check recovery

    Thread-safe for use in async context.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._total_rejected = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"[CircuitBreaker] Initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Whether circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Whether circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    async def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        async with self._lock:
            time_until_retry = None
            if self._state == CircuitState.OPEN and self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                remaining = self.config.recovery_timeout - elapsed
                time_until_retry = max(0, remaining)

            return CircuitBreakerStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=datetime.fromtimestamp(self._last_failure_time) if self._last_failure_time else None,
                last_success_time=datetime.fromtimestamp(self._last_success_time) if self._last_success_time else None,
                total_rejected=self._total_rejected,
                time_until_retry=time_until_retry,
            )

    async def can_execute(self) -> Tuple[bool, Optional[str]]:
        """
        Check if a request can be executed.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True, None

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        # Transition to half-open
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info("[CircuitBreaker] Transitioning to HALF_OPEN state")
                        return True, None

                # Still in open state
                self._total_rejected += 1
                remaining = self.config.recovery_timeout - (time.time() - (self._last_failure_time or time.time()))
                return False, f"Circuit breaker open. Retry in {int(remaining)}s"

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True, None
                else:
                    self._total_rejected += 1
                    return False, "Circuit breaker half-open, test request in progress"

        return False, "Unknown circuit state"

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._last_success_time = time.time()
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open means service recovered
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("[CircuitBreaker] Service recovered, circuit CLOSED")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            error_msg = str(error) if error else "Unknown error"
            logger.warning(
                f"[CircuitBreaker] Failure recorded ({self._failure_count}/{self.config.failure_threshold}): "
                f"{error_msg[:100]}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open means service still broken
                self._state = CircuitState.OPEN
                logger.warning("[CircuitBreaker] Test request failed, circuit remains OPEN")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        f"[CircuitBreaker] Failure threshold reached ({self._failure_count}), "
                        f"circuit OPEN for {self.config.recovery_timeout}s"
                    )

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info("[CircuitBreaker] Manually reset to CLOSED state")


# =============================================================================
# CONCURRENCY LIMITER
# =============================================================================

@dataclass
class ConcurrencyStats:
    """Statistics for concurrency limiter."""
    max_concurrent: int
    current_active: int
    waiting: int
    total_rejected: int
    total_processed: int


class ConcurrencyLimiter:
    """
    Limits concurrent steering requests to prevent GPU memory exhaustion.

    Uses a semaphore to ensure only N requests run simultaneously.
    Excess requests are rejected with a clear message (no queuing).

    Design rationale: Rejection over queuing because:
    - Steering is interactive, users expect immediate feedback
    - Queuing could lead to memory growth and timeout issues
    - Users can easily retry
    """

    def __init__(self, max_concurrent: int = 1):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum simultaneous requests (default: 1 for GPU safety)
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._waiting_count = 0
        self._total_rejected = 0
        self._total_processed = 0
        self._lock = asyncio.Lock()

        logger.info(f"[ConcurrencyLimiter] Initialized: max_concurrent={max_concurrent}")

    async def try_acquire(self) -> Tuple[bool, Optional[str]]:
        """
        Try to acquire a slot for execution.

        Returns:
            Tuple of (acquired: bool, reason: Optional[str])
        """
        # Non-blocking check first
        if self._semaphore.locked():
            async with self._lock:
                self._total_rejected += 1
            return False, "Steering operation in progress. Please wait and try again."

        # Try to acquire
        try:
            acquired = self._semaphore.locked() is False
            if acquired:
                await self._semaphore.acquire()
                async with self._lock:
                    self._active_count += 1
                return True, None
            else:
                async with self._lock:
                    self._total_rejected += 1
                return False, "Steering operation in progress. Please wait and try again."
        except Exception as e:
            return False, f"Failed to acquire lock: {e}"

    async def release(self) -> None:
        """Release a slot after execution completes."""
        self._semaphore.release()
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_processed += 1

    async def get_stats(self) -> ConcurrencyStats:
        """Get current concurrency statistics."""
        async with self._lock:
            return ConcurrencyStats(
                max_concurrent=self.max_concurrent,
                current_active=self._active_count,
                waiting=self._waiting_count,
                total_rejected=self._total_rejected,
                total_processed=self._total_processed,
            )

    @property
    def is_busy(self) -> bool:
        """Check if at capacity."""
        return self._semaphore.locked()


# =============================================================================
# PROCESS ISOLATION (Lightweight approach)
# =============================================================================

class ProcessIsolationManager:
    """
    Manages process isolation for GPU operations.

    Instead of full ProcessPoolExecutor (which would require reloading models
    for each request), this provides:

    1. Watchdog timeout - kills stuck operations
    2. Graceful cleanup - ensures GPU memory is released on failure
    3. Error isolation - captures errors without crashing main process

    For full process isolation with model persistence, consider using
    a dedicated GPU worker service (separate process/container).
    """

    def __init__(self, default_timeout: float = 120.0):
        """
        Initialize process isolation manager.

        Args:
            default_timeout: Default timeout for operations in seconds
        """
        self.default_timeout = default_timeout
        self._operation_count = 0
        self._timeout_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

        logger.info(f"[ProcessIsolation] Initialized: default_timeout={default_timeout}s")

    async def execute_with_isolation(
        self,
        operation: Callable,
        *args,
        timeout: Optional[float] = None,
        cleanup_fn: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute an operation with timeout and error isolation.

        Args:
            operation: Async callable to execute
            timeout: Operation timeout (uses default if not specified)
            cleanup_fn: Optional cleanup function to call on error
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation

        Raises:
            asyncio.TimeoutError: If operation times out
            Exception: If operation fails (after cleanup)
        """
        timeout = timeout or self.default_timeout

        async with self._lock:
            self._operation_count += 1

        try:
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=timeout
            )
            return result

        except asyncio.TimeoutError:
            async with self._lock:
                self._timeout_count += 1

            logger.error(f"[ProcessIsolation] Operation timed out after {timeout}s")

            # Run cleanup if provided
            if cleanup_fn:
                try:
                    cleanup_fn()
                except Exception as cleanup_error:
                    logger.warning(f"[ProcessIsolation] Cleanup error: {cleanup_error}")

            raise

        except Exception as e:
            async with self._lock:
                self._error_count += 1

            logger.error(f"[ProcessIsolation] Operation failed: {e}")

            # Run cleanup if provided
            if cleanup_fn:
                try:
                    cleanup_fn()
                except Exception as cleanup_error:
                    logger.warning(f"[ProcessIsolation] Cleanup error: {cleanup_error}")

            raise

    async def get_stats(self) -> Dict[str, Any]:
        """Get isolation manager statistics."""
        async with self._lock:
            return {
                "total_operations": self._operation_count,
                "timeout_count": self._timeout_count,
                "error_count": self._error_count,
                "default_timeout_seconds": self.default_timeout,
            }


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global instances - created lazily
_circuit_breaker: Optional[CircuitBreaker] = None
_concurrency_limiter: Optional[ConcurrencyLimiter] = None
_process_isolation: Optional[ProcessIsolationManager] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=1,
        ))
    return _circuit_breaker


def get_concurrency_limiter() -> ConcurrencyLimiter:
    """Get or create global concurrency limiter instance."""
    global _concurrency_limiter
    if _concurrency_limiter is None:
        _concurrency_limiter = ConcurrencyLimiter(max_concurrent=1)
    return _concurrency_limiter


def get_process_isolation() -> ProcessIsolationManager:
    """Get or create global process isolation manager instance."""
    global _process_isolation
    if _process_isolation is None:
        from ..core.config import settings
        _process_isolation = ProcessIsolationManager(
            default_timeout=settings.steering_timeout_seconds
        )
    return _process_isolation


async def get_resilience_status() -> Dict[str, Any]:
    """
    Get combined status of all resilience components.

    Returns:
        Dictionary with circuit breaker, concurrency, and isolation stats
    """
    cb = get_circuit_breaker()
    cl = get_concurrency_limiter()
    pi = get_process_isolation()

    cb_stats = await cb.get_stats()
    cl_stats = await cl.get_stats()
    pi_stats = await pi.get_stats()

    return {
        "circuit_breaker": {
            "state": cb_stats.state.value,
            "failure_count": cb_stats.failure_count,
            "success_count": cb_stats.success_count,
            "total_rejected": cb_stats.total_rejected,
            "last_failure": cb_stats.last_failure_time.isoformat() if cb_stats.last_failure_time else None,
            "time_until_retry": cb_stats.time_until_retry,
        },
        "concurrency": {
            "max_concurrent": cl_stats.max_concurrent,
            "current_active": cl_stats.current_active,
            "total_rejected": cl_stats.total_rejected,
            "total_processed": cl_stats.total_processed,
            "is_busy": cl.is_busy,
        },
        "process_isolation": pi_stats,
    }


async def reset_resilience() -> Dict[str, str]:
    """
    Reset all resilience components to initial state.

    Returns:
        Dictionary with reset confirmation messages
    """
    cb = get_circuit_breaker()
    await cb.reset()

    return {
        "circuit_breaker": "Reset to CLOSED state",
        "concurrency": "No reset needed (stateless)",
        "process_isolation": "Counters preserved for monitoring",
    }
