"""
oLLM Server - OpenAI-Compatible API Wrapper for oLLM

This service provides an OpenAI-compatible REST API wrapper around the oLLM library,
enabling drop-in replacement for Ollama with memory-efficient inference capabilities.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Streaming support for real-time responses
- Memory-efficient inference with SSD offloading
- Dynamic model loading and management
- Health checks and model listing endpoints
"""

__version__ = "0.1.0"
