"""Offline reasoning-label generation (issue #98, train-only).

Generates the horizon-aware, action-relevant reasoning labels that supervise
the runtime ``HorizonReasoningHead``. Everything here runs OFFLINE during
preprocessing (a Flyte Processing task, a local script, or a batch job) and is
consumed by training as frozen, versioned artifacts — never called from the
model forward pass or the vehicle.

The teacher backend is model-agnostic (OpenAI-compatible endpoint / mock /
cached), so the same pipeline drives Cosmos3-Nano on vLLM, a Qwen server, or a
local stub with no code change. The taxonomy contract is shared with the
runtime student via ``model_components.reasoning.reasoning_taxonomy`` (which
carries no teacher dependency).
"""
