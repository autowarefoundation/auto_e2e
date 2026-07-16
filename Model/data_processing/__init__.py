"""Offline data-processing utilities (train-only, never on the runtime path).

Unlike ``model_components`` (which must stay runtime-safe), packages under
``data_processing`` are offline preprocessing tools: teacher label generation,
artifact writers, and Flyte tasks. They may import heavy/optional dependencies
and call external endpoints, because they run during preprocessing, never in
the vehicle inference loop or the training forward pass.
"""
