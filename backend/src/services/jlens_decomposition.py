"""Sparse decomposition by gradient pursuit — readout mode (c) (BR-008, BR-009; 022 task 5.3).

Solve for a sparse NON-NEGATIVE combination of at most ``k`` lens directions that best
reconstructs a residual ``h`` at one (layer, position). A lens direction for token ``t``
lives in the residual stream as ``d_t = J^T W_U[t]`` (``J = I`` for the logit lens).

THIS IS NOT TOP-K BY INNER PRODUCT. On an overcomplete, correlated frame the k atoms most
aligned with ``h`` are often near-duplicates of ONE component: ranking by correlation
spends the whole budget re-explaining it and never reaches the second. Pursuit selects an
atom, removes what it explains from the residual, and selects again against what is
LEFT — so a redundant atom stops looking attractive once its twin is in. FPRD §3.3 forbids
the substitution; `test_jlens_decomposition.py` pins that the two disagree on a correlated
frame (022 task 5.6).

THE DICTIONARY IS NEVER FORMED (022 task 6.2). Correlations are ``W_U @ (J r)`` — one
transport and one matvec over the unembedding the readout already holds; synthesis is
``J^T (W_U[S]^T c)`` over the ACTIVE set only. Atom norms (pursuit needs unit atoms, or a
long atom wins by length) are computed in row chunks, so peak memory is
``chunk x d_model``, never ``n_vocab x d_model``.

REPRODUCIBILITY IS PROVENANCE, NOT MATHEMATICS (BR-009). The solution is non-unique by
construction, so every result records ``k``, the solver and its parameters, the iteration
count, the convergence criterion and the CONTROL SEED. The control is a size-matched random
direction (same norm as ``h``, Gaussian, seeded) decomposed by the same solver, and the
explained-variance figure is reported as an EXCESS over it — on a dictionary this
overcomplete a random vector is also "explained" to a degree, and a bare FVE would credit
the lens with that.

The residual is decomposed BEFORE the final norm: the norm is nonlinear and the
decomposition is a linear model of the residual stream. The ranked readout applies the
norm; the two answer different questions and are recorded as different modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

SOLVER_ID = "nonnegative_gradient_pursuit/v1"
CONTROL_CONSTRUCTION = "gaussian_direction_rescaled_to_norm_of_h/torch.Generator(seed)"

#: Rows per chunk when computing atom norms. 4096 x d_model fp32 at d_model 4096 is 64 MiB.
NORM_CHUNK_ROWS = 4096

#: Atoms whose norm is below this are degenerate (an all-zero unembedding row) and are
#: never selected — dividing by ~0 would make them the most "correlated" atom in the frame.
MIN_ATOM_NORM = 1e-8


@dataclass
class LensDictionary:
    """The implicit dictionary ``D[t] = J^T W_U[t] / ||J^T W_U[t]||`` at one layer."""

    unembedding: torch.Tensor  # [n_vocab, d_model], fp32, CPU
    transport: Any  # LensTransport
    layer: int
    norms: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.unembedding = self.unembedding.to(torch.float32)
        self.norms = self._atom_norms()

    def _atom_norms(self) -> torch.Tensor:
        n_vocab = self.unembedding.shape[0]
        out = torch.empty(n_vocab, dtype=torch.float32)
        if getattr(self.transport, "lens_type", None) == "LOGIT_LENS":
            # J = I: the atom IS the unembedding row.
            return self.unembedding.norm(dim=1)
        for start in range(0, n_vocab, NORM_CHUNK_ROWS):
            rows = self.unembedding[start:start + NORM_CHUNK_ROWS]  # [c, d]
            # J^T w for each row w is (w @ J); for a row block that is rows @ J,
            # obtained through the transport's own adjoint, one row at a time
            # would be d_model^2 per row anyway, so batch via pullback on the
            # TRANSPOSE: pullback(rows.T) = J^T rows.T -> [d, c].
            pulled = self.transport.pullback(rows.T.contiguous(), self.layer)  # [d, c]
            out[start:start + rows.shape[0]] = pulled.norm(dim=0)
        return out

    def correlate(self, r: torch.Tensor) -> torch.Tensor:
        """``D r`` for every atom: ``(W_U @ J r) / norms``. Degenerate atoms read -inf."""
        raw = self.unembedding @ self.transport.apply(r, self.layer).to(torch.float32)
        scores = raw / self.norms.clamp_min(MIN_ATOM_NORM)
        return torch.where(self.norms > MIN_ATOM_NORM, scores, torch.full_like(scores, -float("inf")))

    def synthesize(self, active: Sequence[int], coefficients: torch.Tensor) -> torch.Tensor:
        """``sum_t c_t D[t]`` over the active set only."""
        if len(active) == 0:
            return torch.zeros(self.unembedding.shape[1], dtype=torch.float32)
        idx = torch.as_tensor(list(active), dtype=torch.long)
        weights = coefficients / self.norms[idx]  # fold the unit-norm scaling into c
        combo = self.unembedding[idx].T @ weights  # [d_model], in readout space
        return self.transport.pullback(combo, self.layer).to(torch.float32)


@dataclass
class PursuitResult:
    active: List[int]
    coefficients: List[float]
    residual_norm: float
    fve: float
    iterations: int
    converged: bool


def gradient_pursuit(
    h: torch.Tensor,
    dictionary: LensDictionary,
    k: int,
    *,
    max_iterations: Optional[int] = None,
    tolerance: float = 1e-6,
) -> PursuitResult:
    """Non-negative gradient pursuit (Blumensath & Davies 2008), at most ``k`` atoms.

    Each iteration: if the active set has room, admit the atom with the largest POSITIVE
    correlation to the current residual; then take one exact line-search step along the
    gradient restricted to the active set, clip negative coefficients to zero (and drop
    those atoms), and recompute the residual. Stops when no admissible atom correlates
    positively, when the relative residual improvement falls below ``tolerance``, or at
    ``max_iterations`` (default ``4 k``).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    h = h.to(torch.float32).flatten()
    max_iterations = max_iterations if max_iterations is not None else 4 * k
    h_norm_sq = float(h @ h)
    if h_norm_sq == 0.0:
        return PursuitResult([], [], 0.0, 0.0, 0, True)

    active: List[int] = []
    coef = torch.zeros(0, dtype=torch.float32)
    residual = h.clone()
    previous = h_norm_sq
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        correlations = dictionary.correlate(residual)
        if len(active) < k:
            masked = correlations.clone()
            if active:
                masked[torch.as_tensor(active)] = -float("inf")
            best = int(torch.argmax(masked))
            if masked[best] > 0:
                active.append(best)
                coef = torch.cat([coef, torch.zeros(1)])
        if not active:
            converged = True  # nothing correlates positively: the empty set is optimal
            break
        idx = torch.as_tensor(active, dtype=torch.long)
        gradient = correlations[idx]  # D_S r: minus the gradient of 0.5||h - D_S^T c||^2
        # Exact line search along the restricted gradient: step = <r, D_S g> / ||D_S g||^2.
        direction_vec = dictionary.synthesize(active, gradient)
        denom = float(direction_vec @ direction_vec)
        if denom <= 0.0:
            converged = True
            break
        step = float(residual @ direction_vec) / denom
        coef = (coef + step * gradient).clamp(min=0.0)
        keep = coef > 0
        if not bool(keep.all()):
            active = [a for a, kept in zip(active, keep.tolist()) if kept]
            coef = coef[keep]
        residual = h - dictionary.synthesize(active, coef)
        current = float(residual @ residual)
        if previous - current <= tolerance * h_norm_sq:
            # No meaningful improvement. Done if the support cannot grow: it is full, or
            # nothing left correlates positively with what remains.
            if len(active) == k or float(dictionary.correlate(residual).max()) <= 0:
                converged = True
                break
        previous = current

    residual_norm = float(residual.norm())
    fve = 1.0 - (residual_norm ** 2) / h_norm_sq
    order = sorted(range(len(active)), key=lambda i: -float(coef[i]))
    return PursuitResult(
        active=[active[i] for i in order],
        coefficients=[float(coef[i]) for i in order],
        residual_norm=residual_norm,
        fve=fve,
        iterations=iterations,
        converged=converged,
    )


def top_k_by_inner_product(h: torch.Tensor, dictionary: LensDictionary, k: int) -> List[int]:
    """The substitution FPRD §3.3 forbids — kept ONLY so a test can show it differs."""
    correlations = dictionary.correlate(h.to(torch.float32).flatten())
    return [int(i) for i in torch.topk(correlations, k).indices]


def size_matched_control(h: torch.Tensor, seed: int) -> torch.Tensor:
    """A random direction with ``||h||``, reproducible from ``seed`` alone."""
    generator = torch.Generator().manual_seed(int(seed))
    vec = torch.randn(h.numel(), generator=generator, dtype=torch.float32)
    return vec / vec.norm().clamp_min(1e-12) * float(h.to(torch.float32).norm())


def decompose(
    h: torch.Tensor,
    dictionary: LensDictionary,
    *,
    k: int,
    control_seed: int,
    decode: Any,
    position: int,
    max_iterations: Optional[int] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Decompose ``h`` and its size-matched control; return a `DecompositionResult` dict
    plus the excess-FVE figures that are only meaningful against that control."""
    result = gradient_pursuit(h, dictionary, k, max_iterations=max_iterations, tolerance=tolerance)
    control = gradient_pursuit(
        size_matched_control(h, control_seed), dictionary, k,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    return {
        "layer": dictionary.layer,
        "position": position,
        "active_tokens": [decode(i) for i in result.active],
        "active_token_ids": list(result.active),
        "coefficients": result.coefficients,
        "residual_norm": result.residual_norm,
        "fve": result.fve,
        "control_fve": control.fve,
        "excess_fve": result.fve - control.fve,
        "converged": result.converged,
        "provenance": {
            "k": k,
            "solver": SOLVER_ID,
            "solver_params": {
                "max_iterations": max_iterations if max_iterations is not None else 4 * k,
                "tolerance": tolerance,
                "nonnegative": True,
                "unit_norm_atoms": True,
                "pre_final_norm": True,
            },
            "iterations": result.iterations,
            "convergence_criterion": (
                "stop when no unselected atom correlates positively with the residual, or "
                "the support is full and the residual energy improves by <= tolerance x "
                "||h||^2 in one iteration, or max_iterations"
            ),
            "control_seed": int(control_seed),
            "control_construction": CONTROL_CONSTRUCTION,
        },
    }
