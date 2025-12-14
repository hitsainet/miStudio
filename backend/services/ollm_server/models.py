"""
Pydantic models matching OpenAI API schema for compatibility

These models ensure the oLLM server can be used as a drop-in replacement
for any service that uses the OpenAI chat completions API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
import time


# ============================================================================
# Request Models
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in the chat conversation"""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]

    # Generation parameters
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False

    # Advanced parameters
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Not used but accepted for compatibility
    n: Optional[int] = Field(default=1, ge=1)
    seed: Optional[int] = None


# ============================================================================
# Response Models
# ============================================================================

class Usage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """Message in a completion choice"""
    role: str = "assistant"
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Choice(BaseModel):
    """A single completion choice"""
    index: int
    message: ChoiceMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter"]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


# ============================================================================
# Streaming Response Models
# ============================================================================

class DeltaMessage(BaseModel):
    """Delta message for streaming responses"""
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """Choice for streaming responses"""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response"""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


# ============================================================================
# Model Management Models
# ============================================================================

class ModelInfo(BaseModel):
    """Information about an available model"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ollm"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    """Response for listing available models"""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Health & Status Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str
    model_loaded: bool
    current_model: Optional[str] = None
    gpu_available: bool
    memory_usage: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response matching OpenAI format"""
    error: Dict[str, Any]

    @classmethod
    def create(cls, message: str, type: str = "invalid_request_error", code: Optional[str] = None):
        return cls(error={
            "message": message,
            "type": type,
            "code": code
        })
