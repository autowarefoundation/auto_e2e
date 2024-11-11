"""Validation — stdlib only (no jsonschema dependency).

Two layers:
  1. validate_schema(): a small JSON-Schema interpreter covering the subset arch_v1 uses
     (type, enum, const, required, properties, items, minimum/maximum, additionalProperties).
  2. validate_arch_structure(): structural invariants for the arch_v1 IR — edge endpoints
     resolve, group members resolve, the dataflow sub-graph is acyclic (so left-to-right
     layering terminates), with soft notes when inputs/outputs are missing.
"""
