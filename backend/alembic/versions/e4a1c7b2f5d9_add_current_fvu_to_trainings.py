"""add current_fvu to trainings

FVU (Fraction of Variance Unexplained) = var(x - x_hat) / var(x). It is the
metric that actually indicates SAE convergence: 0.0 is perfect reconstruction,
1.0 is no better than predicting the mean. Unlike raw MSE it is scale-free, so
it is comparable across layers and models.

It was already computed per step and persisted to training_metrics.fvu, but the
trainings row carried no live value and the WebSocket payload omitted it — so
the running-training UI could show loss, L0 and dead neurons while the one
convergence signal was invisible.

Nullable by design: only architectures that compute it (JumpReLU) report a
value, and a NULL means "not reported", not "zero".

Revision ID: e4a1c7b2f5d9
Revises: d7e3a91c04b8
"""
from alembic import op
import sqlalchemy as sa

revision = "e4a1c7b2f5d9"
down_revision = "d7e3a91c04b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trainings", sa.Column("current_fvu", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("trainings", "current_fvu")
