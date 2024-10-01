"""Stage 3 — LLM architecture-diagram extraction.

Assembles the context (concrete config + registry-variant guidance + code bundle + AST facts
+ the arch_v1 schema) and invokes the `claude` CLI non-interactively to emit the arch_v1 IR.
Claude is the backbone: it reads __init__ AND forward() and reconstructs the blocks, the
left-to-right data flow, the training-only branches and the losses that rule-based parsing
cannot.

If the `claude` CLI is unavailable, the caller can supply a pre-computed arch IR (e.g. a
checked-in example) via `--arch` so the renderer still runs.