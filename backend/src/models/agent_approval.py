"""
Agent approval request model.

Operator-approval mode for MCP agent steering (Feature 010, FR-3.5): when
``MCP_STEERING_APPROVAL`` is enabled, agent steering calls are stored here as
pending requests instead of being submitted directly. An operator approves or
denies them in the Steering panel; on approval the backend submits the stored
payload itself and records the resulting steering task id.
"""

import uuid
from enum import Enum

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from ..core.database import Base


class ApprovalStatus(str, Enum):
    """Lifecycle of an agent approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class AgentApprovalRequest(Base):
    """A pending/resolved agent steering request awaiting operator action."""

    __tablename__ = "agent_approval_requests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tool_name = Column(String(50), nullable=False)  # steer_compare|steer_sweep|steer_combined
    payload = Column(JSONB, nullable=False)  # full steering request body
    status = Column(String(10), nullable=False, server_default="pending", index=True)
    reason = Column(Text, nullable=True)  # operator's denial reason
    steering_task_id = Column(String(255), nullable=True)  # set on approval
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<AgentApprovalRequest(id={self.id}, tool={self.tool_name}, status={self.status})>"
