"""
SAE Training Framework Defaults

Per-framework configuration registry based on published papers.
Each framework has its own optimizer settings, loss formulation,
and recommended hyperparameters.

Used by:
- Training loop (optimizer betas, epsilon, sparsity warmup behavior)
- Validation (framework-specific checks)
- Frontend (display names, descriptions, field visibility)
"""

from typing import Dict, Any


# Sparsity type determines how sparsity is enforced:
# - "l1": L1 penalty on activations (standard, anthropic, skip, transcoder)
# - "l0": L0 penalty via STE (jumprelu)
# - "topk": Structural sparsity via TopK selection (topk)

FRAMEWORK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "standard_saelens": {
        "display_name": "Standard (SAELens)",
        "paper": "Bricken et al. 2023",
        "description": "Standard SAE with L1 sparsity, ReLU activation, constant_norm_rescale normalization.",
        "optimizer_betas": (0.9, 0.999),
        "adam_epsilon": 1e-8,
        "default_l1_alpha": 5e-4,
        "default_learning_rate": 4e-4,
        "normalize_activations": "constant_norm_rescale",
        "normalize_decoder": True,
        "weight_decay": 0.0,
        "sparsity_type": "l1",
        "sparsity_warmup_steps": 5000,
    },
    "standard_anthropic": {
        "display_name": "Standard (Anthropic)",
        "paper": "Templeton et al. 2024",
        "description": "Standard SAE with Anthropic normalization (E[||x||²]=d_model). L1 coefficient ~5.",
        "optimizer_betas": (0.9, 0.999),
        "adam_epsilon": 1e-8,
        "default_l1_alpha": 5.0,
        "default_learning_rate": 4e-4,
        "normalize_activations": "anthropic_rescale",
        "normalize_decoder": True,
        "weight_decay": 0.0,
        "sparsity_type": "l1",
        "sparsity_warmup_steps": 5000,
    },
    "jumprelu": {
        "display_name": "JumpReLU (Gemma Scope)",
        "paper": "Rajamanoharan et al. 2024",
        "description": "JumpReLU activation with learnable thresholds and L0 penalty via STE.",
        "optimizer_betas": (0.0, 0.999),
        "adam_epsilon": 1e-8,
        "default_sparsity_coeff": 1e-3,
        "default_learning_rate": 7e-5,
        "normalize_activations": "constant_norm_rescale",
        "normalize_decoder": True,
        "weight_decay": 0.0,
        "sparsity_type": "l0",
        "sparsity_warmup_steps": 10000,
    },
    "topk": {
        "display_name": "TopK (OpenAI)",
        "paper": "Gao et al. 2024",
        "description": "Structural sparsity via TopK selection. No sparsity penalty — uses auxiliary dead feature loss.",
        "optimizer_betas": (0.9, 0.999),
        "adam_epsilon": 6.25e-10,
        "default_learning_rate": 3e-4,
        "default_top_k": 64,
        "default_aux_loss_alpha": 1.0 / 32,
        "normalize_activations": "constant_norm_rescale",
        "normalize_decoder": False,
        "weight_decay": 0.0,
        "sparsity_type": "topk",
        "sparsity_warmup_steps": 0,  # No sparsity penalty to warm up
    },
    "skip": {
        "display_name": "Skip",
        "paper": "Community variant",
        "description": "Standard SAE with residual skip connection added to decoder output.",
        "optimizer_betas": (0.9, 0.999),
        "adam_epsilon": 1e-8,
        "default_l1_alpha": 5e-4,
        "default_learning_rate": 4e-4,
        "normalize_activations": "constant_norm_rescale",
        "normalize_decoder": True,
        "weight_decay": 0.0,
        "sparsity_type": "l1",
        "sparsity_warmup_steps": 5000,
    },
    "transcoder": {
        "display_name": "Transcoder",
        "paper": "Dunefsky et al. 2024",
        "description": "Predicts MLP output from MLP input. Uses L1 sparsity on latent features.",
        "optimizer_betas": (0.9, 0.999),
        "adam_epsilon": 1e-8,
        "default_l1_alpha": 5e-4,
        "default_learning_rate": 4e-4,
        "normalize_activations": "constant_norm_rescale",
        "normalize_decoder": True,
        "weight_decay": 0.0,
        "sparsity_type": "l1",
        "sparsity_warmup_steps": 5000,
    },
}


def get_framework_defaults(architecture_type: str) -> Dict[str, Any]:
    """
    Get framework defaults for a given architecture type.

    Falls back to standard_saelens for unknown types (including legacy 'standard').
    """
    # Backward compat
    if architecture_type == "standard":
        architecture_type = "standard_saelens"

    return FRAMEWORK_DEFAULTS.get(architecture_type, FRAMEWORK_DEFAULTS["standard_saelens"])
