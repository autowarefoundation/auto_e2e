"""viz_ml CLI — read PyTorch source, render a left-to-right architecture diagram.

Commands:
  arch     source.py --class Net --config c.json -o net.arch.html [--save-ir net.arch.json]
             Stage 0 (resolve) -> Stage 1 (AST facts) -> Stage 3 (Claude -> arch_v1 IR)
             -> validate + render a self-contained architecture-diagram HTML.
             Use --arch <file.json> to render a pre-computed/hand-edited IR (no Claude call).
  variants source.py --class Net [--config c.json]
             List the registry/factory variants the model can select among.
  facts    source.py --class Net [--config c.json]
             Print the Stage 0/1 code bundle + AST facts (no LLM). For inspection.
  validate net.arch.json
             Validate an arch_v1 IR file against the schema + structural invariants.
"""
