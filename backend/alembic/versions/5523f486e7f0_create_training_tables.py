"""create_training_tables

Revision ID: 5523f486e7f0
Revises: de3c8c763fc1
Create Date: 2025-10-18 12:49:24.073112

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '5523f486e7f0'
down_revision: Union[str, None] = 'de3c8c763fc1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create trainings table
    op.create_table(
        'trainings',
        sa.Column('id', sa.String(255), primary_key=True, comment='Training job ID (format: train_{uuid})'),
        sa.Column('model_id', sa.String(255), sa.ForeignKey('models.id', ondelete='RESTRICT'), nullable=False, comment='Reference to base model'),
        sa.Column('dataset_id', sa.String(255), nullable=False, comment='Reference to training dataset (no FK due to type mismatch)'),
        sa.Column('extraction_id', sa.String(255), sa.ForeignKey('activation_extractions.id', ondelete='RESTRICT'), nullable=True, comment='Reference to activation extraction used for training'),

        # Status and progress (using String instead of Enum for simplicity)
        sa.Column('status', sa.String(50), nullable=False, default='pending', comment='Current training status'),
        sa.Column('progress', sa.Float, nullable=False, default=0.0, comment='Training progress (0-100)'),
        sa.Column('current_step', sa.Integer, nullable=False, default=0, comment='Current training step'),
        sa.Column('total_steps', sa.Integer, nullable=False, comment='Total planned training steps'),

        # Hyperparameters (stored as JSONB for flexibility)
        sa.Column('hyperparameters', postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment='Training hyperparameters'),
        # Contains: hidden_dim, latent_dim, l1_alpha, learning_rate, batch_size,
        #           architecture_type (standard/skip/transcoder), warmup_steps, etc.

        # Current metrics (latest values)
        sa.Column('current_loss', sa.Float, nullable=True, comment='Current reconstruction loss'),
        sa.Column('current_l0_sparsity', sa.Float, nullable=True, comment='Current L0 sparsity (active features)'),
        sa.Column('current_dead_neurons', sa.Integer, nullable=True, comment='Current count of dead neurons'),
        sa.Column('current_learning_rate', sa.Float, nullable=True, comment='Current learning rate'),

        # Error handling
        sa.Column('error_message', sa.Text, nullable=True, comment='Error message if status is failed'),
        sa.Column('error_traceback', sa.Text, nullable=True, comment='Full error traceback for debugging'),

        # Celery task tracking
        sa.Column('celery_task_id', sa.String(255), nullable=True, comment='Celery task ID for this training job'),

        # File paths
        sa.Column('checkpoint_dir', sa.String(1000), nullable=True, comment='Directory containing checkpoints'),
        sa.Column('logs_path', sa.String(1000), nullable=True, comment='Path to training logs file'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Job creation timestamp'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Last update timestamp'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True, comment='Training start timestamp'),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True, comment='Training completion timestamp'),

        sa.PrimaryKeyConstraint('id')
    )

    # Create training_metrics table (time-series data)
    op.create_table(
        'training_metrics',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True, comment='Metric record ID'),
        sa.Column('training_id', sa.String(255), sa.ForeignKey('trainings.id', ondelete='CASCADE'), nullable=False, comment='Reference to training job'),
        sa.Column('step', sa.Integer, nullable=False, comment='Training step number'),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Metric collection timestamp'),

        # Loss metrics
        sa.Column('loss', sa.Float, nullable=False, comment='Total reconstruction loss'),
        sa.Column('loss_reconstructed', sa.Float, nullable=True, comment='Reconstruction component of loss'),
        sa.Column('loss_zero', sa.Float, nullable=True, comment='Zero ablation loss'),

        # Sparsity metrics
        sa.Column('l0_sparsity', sa.Float, nullable=True, comment='L0 sparsity (fraction of active features)'),
        sa.Column('l1_sparsity', sa.Float, nullable=True, comment='L1 sparsity penalty'),
        sa.Column('dead_neurons', sa.Integer, nullable=True, comment='Count of dead neurons at this step'),

        # Training dynamics
        sa.Column('learning_rate', sa.Float, nullable=True, comment='Learning rate at this step'),
        sa.Column('grad_norm', sa.Float, nullable=True, comment='Gradient norm'),

        # Resource metrics
        sa.Column('gpu_memory_used_mb', sa.Float, nullable=True, comment='GPU memory used in MB'),
        sa.Column('samples_per_second', sa.Float, nullable=True, comment='Training throughput'),

        sa.PrimaryKeyConstraint('id')
    )

    # Create checkpoints table
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.String(255), primary_key=True, comment='Checkpoint ID (format: ckpt_{uuid})'),
        sa.Column('training_id', sa.String(255), sa.ForeignKey('trainings.id', ondelete='CASCADE'), nullable=False, comment='Reference to training job'),
        sa.Column('step', sa.Integer, nullable=False, comment='Training step at checkpoint'),

        # Metrics at checkpoint
        sa.Column('loss', sa.Float, nullable=False, comment='Loss at checkpoint'),
        sa.Column('l0_sparsity', sa.Float, nullable=True, comment='L0 sparsity at checkpoint'),

        # File storage
        sa.Column('storage_path', sa.String(1000), nullable=False, comment='Path to checkpoint file (.safetensors)'),
        sa.Column('file_size_bytes', sa.BigInteger, nullable=True, comment='Checkpoint file size in bytes'),

        # Checkpoint metadata
        sa.Column('is_best', sa.Boolean, nullable=False, default=False, comment='True if this is the best checkpoint'),
        sa.Column('extra_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Additional checkpoint metadata'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='Checkpoint creation timestamp'),

        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for efficient queries

    # Trainings indexes
    op.create_index('idx_trainings_status', 'trainings', ['status'])
    op.create_index('idx_trainings_model_id', 'trainings', ['model_id'])
    op.create_index('idx_trainings_dataset_id', 'trainings', ['dataset_id'])
    op.create_index('idx_trainings_extraction_id', 'trainings', ['extraction_id'])
    op.create_index('idx_trainings_created_at', 'trainings', ['created_at'])
    op.create_index('idx_trainings_celery_task', 'trainings', ['celery_task_id'])

    # Training metrics indexes (critical for time-series queries)
    op.create_index('idx_training_metrics_training_id', 'training_metrics', ['training_id'])
    op.create_index('idx_training_metrics_training_step', 'training_metrics', ['training_id', 'step'])
    op.create_index('idx_training_metrics_timestamp', 'training_metrics', ['timestamp'])

    # Checkpoints indexes
    op.create_index('idx_checkpoints_training_id', 'checkpoints', ['training_id'])
    op.create_index('idx_checkpoints_training_step', 'checkpoints', ['training_id', 'step'])
    op.create_index('idx_checkpoints_is_best', 'checkpoints', ['training_id', 'is_best'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_checkpoints_is_best', table_name='checkpoints')
    op.drop_index('idx_checkpoints_training_step', table_name='checkpoints')
    op.drop_index('idx_checkpoints_training_id', table_name='checkpoints')

    op.drop_index('idx_training_metrics_timestamp', table_name='training_metrics')
    op.drop_index('idx_training_metrics_training_step', table_name='training_metrics')
    op.drop_index('idx_training_metrics_training_id', table_name='training_metrics')

    op.drop_index('idx_trainings_celery_task', table_name='trainings')
    op.drop_index('idx_trainings_created_at', table_name='trainings')
    op.drop_index('idx_trainings_extraction_id', table_name='trainings')
    op.drop_index('idx_trainings_dataset_id', table_name='trainings')
    op.drop_index('idx_trainings_model_id', table_name='trainings')
    op.drop_index('idx_trainings_status', table_name='trainings')

    # Drop tables (in reverse order due to foreign keys)
    op.drop_table('checkpoints')
    op.drop_table('training_metrics')
    op.drop_table('trainings')
