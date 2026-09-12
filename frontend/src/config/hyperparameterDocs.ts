/**
 * Comprehensive Hyperparameter Documentation
 *
 * Provides detailed explanations, purpose, and examples for each SAE training hyperparameter.
 * Used by tooltip system to educate users about parameter effects on training outcomes.
 */

export interface HyperparameterDoc {
  name: string;
  purpose: string;
  description: string;
  examples: {
    value: string | number;
    effect: string;
    useCase: string;
  }[];
  recommendations: string[];
  relatedParams?: string[];
  warnings?: string[];
}

export const HYPERPARAMETER_DOCS: Record<string, HyperparameterDoc> = {
  hidden_dim: {
    name: 'Hidden Dimension',
    purpose: 'Defines the size of the model\'s hidden representations that the SAE will learn to encode and decode.',
    description:
      'This is the input/output dimension of your SAE. It must match the hidden state size of the transformer layer you\'re training on. For example, GPT-2 small has 768-dimensional hidden states, while GPT-2 large has 1536.',
    examples: [
      {
        value: 768,
        effect: 'Matches GPT-2 small, BERT-base hidden size',
        useCase: 'Small to medium language models',
      },
      {
        value: 1024,
        effect: 'Matches GPT-2 medium hidden size',
        useCase: 'Medium language models',
      },
      {
        value: 1536,
        effect: 'Matches GPT-2 large hidden size',
        useCase: 'Large language models',
      },
      {
        value: 2048,
        effect: 'Matches LLaMA-7B hidden size',
        useCase: 'Modern large language models',
      },
    ],
    recommendations: [
      'Must exactly match your model\'s hidden state dimension',
      'Check model.config.hidden_size or model.config.d_model',
      'Cannot be changed after training starts',
    ],
    relatedParams: ['latent_dim'],
  },

  latent_dim: {
    name: 'Latent Dimension',
    purpose: 'Controls the number of interpretable features the SAE learns. Larger values allow more fine-grained feature decomposition but increase computational cost.',
    description:
      'This is the SAE width (number of learned features). Research shows optimal ratios are 8-32x the hidden dimension. Larger SAEs learn more monosemantic (single-concept) features but require more memory and training time.',
    examples: [
      {
        value: 4096,
        effect: '~5x expansion: Coarse features, fast training',
        useCase: 'Quick experimentation, proof-of-concept',
      },
      {
        value: 8192,
        effect: '~10x expansion: Balanced feature granularity',
        useCase: 'General-purpose SAE training (hidden_dim=768)',
      },
      {
        value: 16384,
        effect: '~20x expansion: Fine-grained features',
        useCase: 'High-quality interpretability research',
      },
      {
        value: 32768,
        effect: '~40x expansion: Very fine features, slow',
        useCase: 'State-of-art SAE quality (requires lots of memory)',
      },
    ],
    recommendations: [
      'Use 8-16x hidden_dim as a starting point',
      'Anthropic uses 16-32x for high-quality SAEs',
      'Larger SAEs → better interpretability but higher memory cost',
      'Memory usage scales linearly with latent_dim',
    ],
    relatedParams: ['hidden_dim', 'l1_alpha'],
    warnings: [
      'Very large latent_dim (>32x) may not fit in GPU memory',
      'Requires proportionally smaller l1_alpha (validator will warn)',
    ],
  },

  l1_alpha: {
    name: 'L1 Sparsity Coefficient',
    purpose: 'Controls how aggressively the SAE encourages sparse feature activations. This is the most critical hyperparameter for training quality.',
    description:
      'The L1 penalty coefficient (λ) in loss = reconstruction_loss + λ × L1(activations). Higher values force sparser features (fewer active at once) but risk "race to zero" where all features die. Lower values produce dense, polysemantic features.',
    examples: [
      {
        value: 0.0001,
        effect: 'Very weak sparsity → Dense features (L0 > 20%)',
        useCase: 'NOT RECOMMENDED: Features will be polysemantic',
      },
      {
        value: 0.001,
        effect: 'Moderate sparsity → Balanced (L0 ≈ 5-10%)',
        useCase: 'Small SAEs (latent_dim ≈ 4k-8k)',
      },
      {
        value: 0.003,
        effect: 'Good sparsity → Monosemantic features (L0 ≈ 3-5%)',
        useCase: 'Medium SAEs (latent_dim ≈ 8k-16k) - RECOMMENDED',
      },
      {
        value: 0.01,
        effect: 'Strong sparsity → Very sparse (L0 ≈ 1-3%)',
        useCase: 'Large SAEs (latent_dim ≈ 32k+)',
      },
      {
        value: 0.1,
        effect: '🚨 TOO HIGH → "Race to zero" collapse',
        useCase: 'CRITICAL ERROR: All features will die!',
      },
    ],
    recommendations: [
      'Use formula: 0.01 / sqrt(latent_dim / 8192)',
      'For latent_dim=8192: use l1_alpha ≈ 0.01',
      'For latent_dim=16384: use l1_alpha ≈ 0.007',
      'Monitor L0 during training: target 1-5% activation',
      'If L0 drops dramatically (>70%) → decrease l1_alpha immediately',
    ],
    relatedParams: ['target_l0', 'latent_dim'],
    warnings: [
      '🚨 CRITICAL: Values >5x recommended will cause "race to zero"',
      'Encoder biases drift negative → all features output zeros',
      'Training appears to converge but SAE is useless',
      'Use validator warnings to avoid this catastrophic failure',
    ],
  },

  target_l0: {
    name: 'Target L0 Sparsity',
    purpose: 'Target fraction of features that should be active (non-zero) on average. Guides l1_alpha tuning during training.',
    description:
      'L0 sparsity is the fraction of features that activate (output non-zero values) for typical inputs. Lower L0 = sparser = more monosemantic features. This parameter is informational and used for monitoring, not directly enforced.',
    examples: [
      {
        value: 0.01,
        effect: '1% activation → Extremely sparse features',
        useCase: 'Large SAEs seeking maximum interpretability',
      },
      {
        value: 0.03,
        effect: '3% activation → Very sparse features',
        useCase: 'High-quality interpretability (most research)',
      },
      {
        value: 0.05,
        effect: '5% activation → Moderate sparsity (default)',
        useCase: 'Balanced interpretability and reconstruction',
      },
      {
        value: 0.10,
        effect: '10% activation → Weakly sparse',
        useCase: 'Prioritizing reconstruction over interpretability',
      },
      {
        value: 0.20,
        effect: '20% activation → Dense, polysemantic features',
        useCase: 'NOT RECOMMENDED for interpretability',
      },
    ],
    recommendations: [
      'Default: 0.05 (5% activation rate)',
      'For interpretability: aim for 0.01-0.05',
      'Monitor actual L0 during training via metrics',
      'If actual L0 >> target_l0: increase l1_alpha',
      'If actual L0 << target_l0 or many dead neurons: decrease l1_alpha',
    ],
    relatedParams: ['l1_alpha'],
  },

  top_k_sparsity: {
    name: 'Top-K Active Features',
    purpose: 'GUARANTEES exact sparsity by keeping only the top-K most active features per sample. Replaces L1 penalty with direct sparsity enforcement.',
    description:
      'Enter the number of features (K) to keep active per sample. For example, K=64 means exactly 64 out of your total latent features will fire for each input. L1 penalty is automatically disabled when Top-K is set. An auxiliary loss helps dead features learn by reconstructing the residual.',
    examples: [
      {
        value: 'empty' as any,
        effect: 'Use L1 penalty (soft sparsity)',
        useCase: 'Traditional SAE training with L1 regularization',
      },
      {
        value: 64,
        effect: 'Keep exactly 64 active features per sample (0.39% for 16384 latent_dim)',
        useCase: 'Standard interpretable SAE — recommended starting point',
      },
      {
        value: 32,
        effect: 'Keep exactly 32 active features per sample (very sparse)',
        useCase: 'Maximum interpretability, fewer features explain each input',
      },
      {
        value: 128,
        effect: 'Keep exactly 128 active features per sample (denser)',
        useCase: 'Better reconstruction quality, slightly less interpretable',
      },
    ],
    recommendations: [
      'Leave empty to use L1 penalty (default)',
      'Use Top-K when L1 alpha is difficult to tune',
      'Start with K=64 for 16K features (typical for interpretability research)',
      'L1 penalty is automatically disabled when Top-K is set',
      'Dead features get gradient signal via auxiliary reconstruction loss',
    ],
    warnings: [
      'Top-K is HARD sparsity: gradients only flow to the top-K features',
      'Sparsity level is fixed during training (unlike L1 which adapts)',
      'May have sharper feature boundaries than L1 sparsity',
    ],
    relatedParams: ['l1_alpha', 'target_l0'],
  },

  learning_rate: {
    name: 'Learning Rate',
    purpose: 'Controls the step size for gradient descent optimization. Too high causes instability, too low slows convergence.',
    description:
      'The learning rate (η) determines how much to adjust weights based on gradients. SAE training typically uses Adam optimizer with learning rates 10-100x smaller than typical neural network training.',
    examples: [
      {
        value: 0.00001,
        effect: 'Very slow convergence, stable',
        useCase: 'Very large SAEs or unstable training',
      },
      {
        value: 0.0001,
        effect: 'Slow but steady progress',
        useCase: 'Large SAEs (latent_dim > 32k)',
      },
      {
        value: 0.0003,
        effect: 'Balanced speed and stability (default)',
        useCase: 'Most SAE training (recommended starting point)',
      },
      {
        value: 0.001,
        effect: 'Faster convergence, monitor for instability',
        useCase: 'Small SAEs or when time-constrained',
      },
      {
        value: 0.01,
        effect: 'Risk of divergence or oscillation',
        useCase: 'NOT RECOMMENDED for SAE training',
      },
    ],
    recommendations: [
      'Default: 0.0003 for most SAE training',
      'Use learning rate warmup (see warmup_steps)',
      'If loss oscillates wildly: decrease by 2-5x',
      'If loss decreases very slowly: try increasing by 2x',
      'Adam optimizer adapts per-parameter, so fixed LR is often sufficient',
    ],
    relatedParams: ['warmup_steps', 'weight_decay'],
  },

  batch_size: {
    name: 'Batch Size',
    purpose: 'Number of samples processed together in each training step. Affects memory usage, training stability, and speed.',
    description:
      'Larger batches provide more stable gradient estimates and better GPU utilization but require more memory. SAE training typically uses large batches (1024-8192) for stable sparsity statistics.',
    examples: [
      {
        value: 512,
        effect: 'Small batch: 2-4 GB memory, noisy gradients',
        useCase: 'Memory-constrained environments',
      },
      {
        value: 2048,
        effect: 'Medium batch: 4-8 GB memory, stable',
        useCase: 'Edge devices (Jetson)',
      },
      {
        value: 4096,
        effect: 'Large batch: 8-12 GB memory, very stable (default)',
        useCase: 'High-quality SAE training on modern GPUs',
      },
      {
        value: 8192,
        effect: 'Very large: 16+ GB memory, maximum stability',
        useCase: 'Research-grade SAE training on A100/H100',
      },
    ],
    recommendations: [
      'Default: 4096 for 8-16 GB GPUs',
      'Use largest batch that fits in memory',
      'Larger batches → more stable L0 sparsity estimates',
      'If OOM (out of memory): reduce by half',
      'Consider gradient accumulation for effective larger batches',
    ],
    relatedParams: ['total_steps', 'learning_rate'],
    warnings: [
      'Memory usage scales linearly with batch_size',
      'Check memory estimation panel before training',
    ],
  },

  total_steps: {
    name: 'Total Training Steps',
    purpose: 'Number of gradient descent steps (batches) to train for. More steps allow better convergence but take longer.',
    description:
      'One step = one batch processed. Total samples seen = total_steps × batch_size. SAE training typically requires 10,000-500,000 steps depending on model and data size.',
    examples: [
      {
        value: 10000,
        effect: '~40M samples (batch=4096): Quick experiment',
        useCase: 'Testing hyperparameters, proof-of-concept',
      },
      {
        value: 50000,
        effect: '~200M samples: Reasonable quality SAE',
        useCase: 'Initial training runs, moderate quality',
      },
      {
        value: 100000,
        effect: '~400M samples: Good quality SAE (default)',
        useCase: 'Production SAE training',
      },
      {
        value: 500000,
        effect: '~2B samples: State-of-art quality',
        useCase: 'Research-grade SAEs, publication quality',
      },
    ],
    recommendations: [
      'Default: 100,000 steps (about 1-2 hours on GPU)',
      'Monitor loss curve: stop if loss plateaus early',
      'More data → fewer steps needed for convergence',
      'Smaller models → fewer steps needed',
      'Use checkpoints to resume if interrupted',
    ],
    relatedParams: ['batch_size', 'checkpoint_interval'],
  },

  warmup_steps: {
    name: 'Learning Rate Warmup Steps',
    purpose: 'Gradually increases learning rate from 0 to target value at start of training. Improves stability and prevents early divergence.',
    description:
      'Linear warmup: learning_rate increases from 0 to target over warmup_steps. Prevents large gradient updates before weights are initialized well. Standard practice in transformer training.',
    examples: [
      {
        value: 0,
        effect: 'No warmup: Start at full learning rate',
        useCase: 'Small learning rates (< 0.0001), stable training',
      },
      {
        value: 500,
        effect: 'Short warmup: Gentle start',
        useCase: 'Small SAEs or conservative training',
      },
      {
        value: 1000,
        effect: 'Standard warmup: Good stability (default)',
        useCase: 'Most SAE training scenarios',
      },
      {
        value: 5000,
        effect: 'Long warmup: Very gradual start',
        useCase: 'Large SAEs or high learning rates',
      },
    ],
    recommendations: [
      'Default: 1000 steps (1% of total_steps=100k)',
      'Use 1-5% of total_steps for warmup',
      'If training unstable early: increase warmup_steps',
      'If using small learning_rate: warmup less important',
    ],
    relatedParams: ['learning_rate', 'total_steps', 'sparsity_warmup_steps'],
  },

  sparsity_warmup_steps: {
    name: 'Sparsity Warmup Steps',
    purpose: 'Linearly ramps the sparsity penalty (L1 or L0) from 0 to full value. This is the single most important setting for preventing dead neurons.',
    description:
      'During the first N steps, the sparsity penalty coefficient is scaled by step/N. This allows the SAE to first learn meaningful features before the sparsity constraint pushes them to be selective. Without sparsity warmup, the full penalty from step 0 kills features before they form.',
    examples: [
      {
        value: 0,
        effect: 'No warmup: Full sparsity penalty from step 0',
        useCase: 'Only if very low l1_alpha/sparsity_coeff',
      },
      {
        value: 2000,
        effect: 'Short warmup: Quick ramp to full sparsity',
        useCase: 'Short training runs (< 20k steps)',
      },
      {
        value: 5000,
        effect: 'Standard warmup: Good balance (recommended)',
        useCase: 'Most SAE training scenarios',
      },
      {
        value: 10000,
        effect: 'Long warmup: Very gradual sparsity introduction',
        useCase: 'Large SAEs (64k+ features) or high sparsity coefficients',
      },
    ],
    recommendations: [
      'Default: 5000 steps (critical for preventing dead neurons)',
      'Set to 10% of total_steps for a safe default',
      'JumpReLU benefits even more from sparsity warmup due to L0 penalty',
      'If > 50% dead neurons: increase sparsity_warmup_steps',
    ],
    relatedParams: ['l1_alpha', 'sparsity_coeff', 'warmup_steps'],
  },

  weight_decay: {
    name: 'Weight Decay (L2 Regularization)',
    purpose: 'Adds penalty for large weights, encouraging simpler models and preventing overfitting.',
    description:
      'L2 regularization coefficient. Penalizes sum of squared weights. Less critical for SAE training than for standard neural networks because L1 sparsity already regularizes heavily.',
    examples: [
      {
        value: 0.0,
        effect: 'No weight decay: Let L1 sparsity handle regularization',
        useCase: 'Most SAE training (default)',
      },
      {
        value: 0.001,
        effect: 'Weak decay: Slight preference for smaller weights',
        useCase: 'Large SAEs on limited data',
      },
      {
        value: 0.01,
        effect: 'Moderate decay: Explicit weight control',
        useCase: 'If observing weight magnitude explosions',
      },
    ],
    recommendations: [
      'Default: 0.0 (disabled)',
      'SAE training relies primarily on L1 sparsity, not L2',
      'Only use if encoder weights grow unusually large',
      'If using: start with 0.001 and monitor',
    ],
    relatedParams: ['l1_alpha', 'learning_rate'],
  },

  grad_clip_norm: {
    name: 'Gradient Clipping Norm',
    purpose: 'Clips gradients if their norm exceeds threshold. Prevents exploding gradients and training instability.',
    description:
      'Maximum allowed gradient L2 norm. If gradients exceed this, they are scaled down proportionally. Helps with training stability, especially early in training.',
    examples: [
      {
        value: 0.0,
        effect: 'No clipping: Risk of exploding gradients',
        useCase: 'Very stable training setups only',
      },
      {
        value: 1.0,
        effect: 'Moderate clipping: Good balance (default)',
        useCase: 'Most SAE training scenarios',
      },
      {
        value: 5.0,
        effect: 'Loose clipping: Rarely triggers',
        useCase: 'If loss decreases smoothly without clipping',
      },
    ],
    recommendations: [
      'Default: 1.0',
      'If loss spikes or NaN: decrease to 0.1-0.5',
      'If training very stable: can disable or increase',
      'Monitor clipping frequency in logs',
    ],
    relatedParams: ['learning_rate'],
  },

  checkpoint_interval: {
    name: 'Checkpoint Save Interval',
    purpose: 'How often (in steps) to save model checkpoints. Balances safety against storage usage.',
    description:
      'Saves model weights, optimizer state, and training progress every N steps. Allows resuming training if interrupted and provides snapshots for evaluation.',
    examples: [
      {
        value: 1000,
        effect: 'Frequent saves: 100 checkpoints per 100k steps',
        useCase: 'Unstable hardware, frequent evaluation',
      },
      {
        value: 5000,
        effect: 'Moderate: 20 checkpoints per 100k steps (default)',
        useCase: 'Balanced safety and storage',
      },
      {
        value: 10000,
        effect: 'Infrequent: 10 checkpoints per 100k steps',
        useCase: 'Stable training, limited storage',
      },
    ],
    recommendations: [
      'Default: 5000 (5% of total_steps)',
      'Each checkpoint: ~50-500 MB depending on SAE size',
      'Save more frequently if hardware is unstable',
      'Can evaluate SAE quality at each checkpoint',
    ],
    relatedParams: ['total_steps', 'log_interval'],
  },

  log_interval: {
    name: 'Metrics Logging Interval',
    purpose: 'How often (in steps) to compute and log training metrics. More frequent logging helps monitor training progress but adds overhead.',
    description:
      'Logs loss, L0 sparsity, dead neurons, and other metrics every N steps. Enables real-time monitoring via WebSocket and post-hoc analysis.',
    examples: [
      {
        value: 10,
        effect: 'Very frequent: Fine-grained progress tracking',
        useCase: 'Debugging, hyperparameter tuning',
      },
      {
        value: 100,
        effect: 'Standard: Good balance (default)',
        useCase: 'Most training scenarios',
      },
      {
        value: 1000,
        effect: 'Infrequent: Minimal overhead',
        useCase: 'Very long training runs',
      },
    ],
    recommendations: [
      'Default: 100 (0.1% of total_steps)',
      'More frequent logging adds ~1-5% compute overhead',
      'WebSocket updates happen at this interval',
      'Can always check detailed metrics in database',
    ],
    relatedParams: ['checkpoint_interval'],
  },

  dead_neuron_threshold: {
    name: 'Dead Neuron Detection Threshold',
    purpose: 'Number of steps a neuron must be inactive before considered "dead" for resampling.',
    description:
      'A neuron is "dead" if it never activates (outputs zero) for this many consecutive steps. Dead neurons waste capacity. If resample_dead_neurons enabled, dead neurons are re-initialized.',
    examples: [
      {
        value: 1000,
        effect: 'Aggressive resampling: Quick to revive neurons',
        useCase: 'Small datasets, many dead neurons expected',
      },
      {
        value: 5000,
        effect: 'Moderate: Balance patience vs capacity (default)',
        useCase: 'Most SAE training scenarios',
      },
      {
        value: 10000,
        effect: 'Conservative: Give neurons time to activate',
        useCase: 'Large datasets, patient training',
      },
    ],
    recommendations: [
      'Default: 10,000 steps',
      'Set to 5-10% of total_steps',
      'If many dead neurons persist: decrease l1_alpha',
      'Resampling helps but proper l1_alpha is more important',
    ],
    relatedParams: ['resample_dead_neurons', 'l1_alpha'],
  },

  resample_dead_neurons: {
    name: 'Resample Dead Neurons',
    purpose: 'Whether to re-initialize dead neurons during training. Helps recover wasted capacity.',
    description:
      'When enabled, neurons that remain dead for dead_neuron_threshold steps are re-initialized with new random weights. This can help recover capacity but is not a substitute for proper l1_alpha tuning.',
    examples: [
      {
        value: 'true',
        effect: 'Enabled: Periodically revive dead neurons (default)',
        useCase: 'Most training scenarios',
      },
      {
        value: 'false',
        effect: 'Disabled: Let dead neurons stay dead',
        useCase: 'Studying neuron death patterns, research',
      },
    ],
    recommendations: [
      'Default: Enabled (true)',
      'Helps when l1_alpha slightly too high',
      'Not a fix for "race to zero" (all neurons dead)',
      'Monitor dead neuron count in training metrics',
      'If >50% dead: decrease l1_alpha immediately',
    ],
    relatedParams: ['dead_neuron_threshold', 'l1_alpha', 'resample_interval'],
    warnings: [
      'Resampling cannot fix catastrophic "race to zero"',
      'If all neurons dying: l1_alpha is dangerously high',
    ],
  },

  resample_interval: {
    name: 'Resample Interval',
    purpose: 'How often (in training steps) to check for and resample dead neurons. Balances responsiveness vs stability.',
    description:
      'Dead neurons are re-initialized every resample_interval steps if they remain dead for dead_neuron_threshold steps. Too frequent: instability from constant resampling. Too infrequent: dead neurons waste capacity for longer.',
    examples: [
      {
        value: 2500,
        effect: 'Aggressive: Frequent resampling',
        useCase: 'High l1_alpha, many neurons dying',
      },
      {
        value: 5000,
        effect: 'Balanced: Good trade-off (default)',
        useCase: 'Most training scenarios',
      },
      {
        value: 10000,
        effect: 'Conservative: Rare resampling',
        useCase: 'Low learning_rate, stable training',
      },
      {
        value: 15000,
        effect: 'Very conservative: Maximum stability',
        useCase: 'Fine-tuning, avoiding disruption',
      },
    ],
    recommendations: [
      'Default: 5000 steps',
      'If sparsity oscillates wildly: increase interval (10k-15k)',
      'If too many dead neurons accumulate: decrease interval (2.5k)',
      'Should be >> dead_neuron_threshold for stability',
      'Pair with lower learning_rate if increasing interval',
    ],
    relatedParams: ['resample_dead_neurons', 'dead_neuron_threshold', 'learning_rate'],
    warnings: [
      'Too frequent resampling + high learning_rate = wild oscillations',
      'Newly resampled neurons need time to stabilize',
      'Resampling disrupts training dynamics temporarily',
    ],
  },

  architecture_type: {
    name: 'SAE Architecture Type',
    purpose: 'Selects the SAE architecture variant. Different architectures have different trade-offs.',
    description:
      'Standard: Classic SAE with encoder and decoder. Skip: Adds residual connections for better reconstruction. Transcoder: Layer-to-layer mapping for cross-layer features. JumpReLU: Gemma Scope architecture with learnable per-feature thresholds.',
    examples: [
      {
        value: 'standard',
        effect: 'x → ReLU(W_enc·x + b) → W_dec·z + b → x̂',
        useCase: 'Default: Proven architecture, best studied',
      },
      {
        value: 'skip',
        effect: 'x̂ = x + W_dec·ReLU(W_enc·x + b): Residual connection',
        useCase: 'Better reconstruction, harder to interpret',
      },
      {
        value: 'transcoder',
        effect: 'x_i → ReLU(W_enc·x_i + b) → W_dec·z → x_j',
        useCase: 'Maps layer i → layer j (cross-layer features)',
      },
      {
        value: 'jumprelu',
        effect: 'z → JumpReLU_θ(z) = z ⊙ H(z - θ): Learnable thresholds',
        useCase: 'Gemma Scope: Per-feature learnable thresholds for optimal sparsity',
      },
    ],
    recommendations: [
      'Default: Standard (most research uses this)',
      'JumpReLU: State-of-art from Gemma Scope paper, uses L0 loss instead of L1',
      'Skip: Use if reconstruction quality is priority',
      'Transcoder: Advanced use case, study layer relationships',
    ],
    relatedParams: ['hidden_dim', 'latent_dim', 'sparsity_coeff'],
  },

  // JumpReLU-specific parameters (Gemma Scope architecture)
  initial_threshold: {
    name: 'Initial JumpReLU Threshold',
    purpose: 'Sets the starting threshold value for JumpReLU activation. Each feature learns its own optimal threshold during training.',
    description:
      'JumpReLU uses a learnable per-feature threshold θ: JumpReLU_θ(z) = z ⊙ H(z - θ), where H is the Heaviside step function. The initial_threshold sets the starting point for all thresholds, which then adapt during training to achieve optimal sparsity per feature.',
    examples: [
      {
        value: 0.1,
        effect: 'Low threshold → Most features start active (~30-40% L0)',
        useCase: 'When you want very gradual sparsification',
      },
      {
        value: 0.5,
        effect: 'Default threshold → Matches pre-activation scale with constant_norm_rescale (recommended)',
        useCase: 'Standard JumpReLU training — starts at ~15% L0, thresholds adapt to target',
      },
      {
        value: 1.0,
        effect: 'Higher threshold → Features start sparse (~5% L0)',
        useCase: 'When pre-activations have larger magnitude',
      },
    ],
    recommendations: [
      'Default: 0.5 (matches typical pre-activation magnitude with constant_norm_rescale)',
      'Should be close to the expected pre-activation scale for good gradient flow',
      'Too small (e.g. 0.001) causes gradient bottleneck — thresholds cannot move in log-space',
      'Thresholds adapt during training via STE gradient descent',
    ],
    relatedParams: ['bandwidth', 'sparsity_coeff', 'architecture_type'],
  },

  bandwidth: {
    name: 'KDE Bandwidth (ε)',
    purpose: 'Controls the smoothness of gradient estimation for the JumpReLU threshold. Used in the Straight-Through Estimator (STE) for backpropagation.',
    description:
      'Since JumpReLU uses a step function (H), gradients would be zero almost everywhere. The bandwidth ε defines a Gaussian kernel K_ε(z - θ) for smooth gradient estimation. Smaller ε → sharper gradients near threshold. Larger ε → smoother gradients over wider range.',
    examples: [
      {
        value: 0.001,
        effect: 'Sharp gradients → Only features very close to threshold get gradient',
        useCase: 'Fine-grained threshold adjustment (may be too narrow for initial training)',
      },
      {
        value: 0.01,
        effect: 'Default smoothness → Good gradient coverage across features (recommended)',
        useCase: 'Standard JumpReLU training — covers ±0.03 around threshold',
      },
      {
        value: 0.05,
        effect: 'Very smooth → Wide gradient window, more features updated per step',
        useCase: 'When training is unstable or thresholds are slow to converge',
      },
    ],
    recommendations: [
      'Default: 0.01 — gives gradient coverage across ~6% of pre-activation range',
      'Smaller ε → fewer features get gradient per step (slower threshold learning)',
      'Larger ε → more features get gradient (faster but less precise)',
      'Must be proportional to pre-activation scale for effective gradient flow',
    ],
    relatedParams: ['initial_threshold', 'sparsity_coeff'],
    warnings: [
      'Very small bandwidth (< 0.001) causes gradient bottleneck — thresholds cannot move',
      'Very large bandwidth (> 0.1) may prevent thresholds from converging precisely',
    ],
  },

  sparsity_coeff: {
    name: 'L0 Sparsity Coefficient (λ)',
    purpose: 'Controls the strength of the L0 sparsity penalty in JumpReLU. This is the key sparsity hyperparameter for JumpReLU (replaces l1_alpha).',
    description:
      'JumpReLU uses L0 loss per Gemma Scope paper: L = E[||x - x̂||² + λ · Σᵢ H(zᵢ - θᵢ)], where the L0 term is the count of active features per sample. λ controls the sparsity-reconstruction tradeoff.',
    examples: [
      {
        value: 0.0003,
        effect: 'Weak L0 penalty → More features active',
        useCase: 'Prioritizing reconstruction over sparsity',
      },
      {
        value: 0.001,
        effect: 'Default → Balanced sparsity',
        useCase: 'Standard JumpReLU training',
      },
      {
        value: 0.005,
        effect: 'Strong L0 penalty → Fewer active features',
        useCase: 'When you need very sparse, interpretable features',
      },
    ],
    recommendations: [
      'Default: 1e-3 (Gemma Scope paper uses 6e-4)',
      'At L0=50 active features: loss_l0 = 1e-3 × 50 = 0.05',
      'Typical range: 1e-4 to 5e-3',
      'This replaces l1_alpha for JumpReLU architecture',
      'Lower values → more features active → better reconstruction',
      'Higher values → fewer features active → better interpretability',
    ],
    relatedParams: ['initial_threshold', 'l1_alpha', 'target_l0'],
    warnings: [
      'Coefficient scale depends on latent_dim and model — tune empirically',
      'L0 uses STE (Straight-Through Estimator) for gradient flow through Heaviside',
    ],
  },

  normalize_decoder: {
    name: 'Normalize Decoder Columns',
    purpose: 'Whether to normalize decoder columns to unit norm after each training step. Required for proper JumpReLU training.',
    description:
      'In JumpReLU SAEs, decoder columns (features) are constrained to unit norm. This ensures that the magnitude of each feature\'s contribution is controlled by the encoder, not the decoder. Gradients are projected orthogonal to decoder columns to preserve this constraint.',
    examples: [
      {
        value: 'true',
        effect: 'Decoder columns normalized to ||W_dec[:,i]|| = 1',
        useCase: 'Required for JumpReLU (default and recommended)',
      },
      {
        value: 'false',
        effect: 'Decoder columns can have arbitrary norm',
        useCase: 'NOT RECOMMENDED for JumpReLU',
      },
    ],
    recommendations: [
      'Default: true (always use for JumpReLU)',
      'Required per Gemma Scope paper',
      'Ensures interpretable feature directions',
      'Decoder gradients are projected orthogonal to columns',
    ],
    relatedParams: ['architecture_type', 'sparsity_coeff'],
    warnings: [
      'Disabling this for JumpReLU will break training',
      'Decoder norm explosion may occur without normalization',
    ],
  },
};

/**
 * Get documentation for a specific hyperparameter
 */
export function getHyperparameterDoc(paramName: string): HyperparameterDoc | undefined {
  return HYPERPARAMETER_DOCS[paramName];
}

/**
 * Get all hyperparameter names with documentation
 */
export function getAllHyperparameterNames(): string[] {
  return Object.keys(HYPERPARAMETER_DOCS);
}
