"""
Steering Resilience Infrastructure.

Provides a circuit breaker for the steering service, so repeated failures stop
new dispatches instead of piling up behind a wedged GPU.

MIS-E2E-062 — THIS LAYER WAS ENTIRELY UNREACHABLE.

Not one state-mutating function had a caller. `_failure_count` starts at 0 and
is incremented only in `record_failure`; `_state` starts CLOSED and leaves it
only inside the same dead function. So `GET /steering/status`, which computes
`"healthy" if circuit_breaker.state == "closed"`, could only ever return
"healthy" — no matter how many steering tasks had failed — and `POST
/steering/reset` reset state that was never non-default. The endpoint's own
docstring says "use this to monitor steering health and diagnose issues": an
operator diagnosing a steering outage was told, by construction, always, that
the service was fine.

Resolved by SPLITTING the module rather than wiring all of it:

  * `CircuitBreaker` is now WIRED. `can_execute()` gates the three async
    dispatch endpoints and `record_success` / `record_failure` are driven from
    `GET /async/result/{task_id}`, which is where the API first learns a task's
    outcome. Both live in the API process, so the state the endpoint reports is
    the state that was actually recorded.

  * `ConcurrencyLimiter` and `ProcessIsolationManager` are DELETED. They hold a
    semaphore and run an operation in-process with a timeout, which cannot
    apply to a fire-and-forget `apply_async`: there is no in-process operation
    to bound, and GPU serialisation is already the Celery worker's concurrency
    setting. Wiring them would have produced a second constant.

The rule from CLAUDE.md — "a capability is not shipped until a test FAILS when
its wiring is removed" — could not be applied to any of this before, because
nothing was wired. `tests/unit/test_steering_resilience_wired.py` applies it now.
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

# ConcurrencyLimiter, ConcurrencyStats and ProcessIsolationManager were DELETED
# here (MIS-E2E-062). See the module docstring: a semaphore and an in-process
# timeout cannot bound a fire-and-forget `apply_async`, and GPU serialisation is
# already the Celery worker's concurrency setting. They had zero callers and zero
# tests. Wiring them would have produced a second always-healthy constant.



# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global instances - created lazily
_circuit_breaker: Optional[CircuitBreaker] = None


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




async def get_resilience_status() -> Dict[str, Any]:
    """
    Get combined status of all resilience components.

    Returns:
        Dictionary with circuit breaker, concurrency, and isolation stats
    """
    cb = get_circuit_breaker()
    cb_stats = await cb.get_stats()

    return {
        "circuit_breaker": {
            "state": cb_stats.state.value,
            "failure_count": cb_stats.failure_count,
            "success_count": cb_stats.success_count,
            "total_rejected": cb_stats.total_rejected,
            "last_failure": (
                cb_stats.last_failure_time.isoformat()
                if cb_stats.last_failure_time
                else None
            ),
            "time_until_retry": cb_stats.time_until_retry,
            # Honest about the scope of the observation. The breaker is
            # in-process: it sees the dispatches and outcomes THIS API process
            # handled. That is coherent at the deployed shape (one replica, one
            # uvicorn worker) and would need shared state behind `--workers`.
            "scope": "api-process",
        },
    }


async def reset_resilience() -> Dict[str, str]:
    """
    Reset all resilience components to initial state.

    Returns:
        Dictionary with reset confirmation messages
    """
    cb = get_circuit_breaker()
    await cb.reset()

    return {"circuit_breaker": "Reset to CLOSED state"}
