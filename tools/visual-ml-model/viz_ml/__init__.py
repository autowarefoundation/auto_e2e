"""visual-ml-model: read PyTorch model source, render a left-to-right architecture diagram.

Produces a paper/README-style architecture figure (multiple inputs on the left, the data
spine through the middle, outputs + a loss column on the right, branching/merging, dashed
loss/feedback edges, and an "ONLY DURING TRAINING" band) as a self-contained, dependency-free
HTML/SVG file.

Pipeline (see README):
  Stage 0  resolve  — locate the target class, follow same-repo imports (incl. registry/
                      factory variant selection), build a code bundle
  Stage 1  ast      — stdlib `ast` facts: submodule inventory, register_buffer persistence,
                      forward() skeleton (no torch import)
  Stage 3  extract  — Claude reads the bundle + facts + config and emits an arch_v1 IR JSON
  Stage 4  validate — stdlib-only schema + structural-invariant checks
           render   — deterministic left-to-right layout -> inline SVG -> self-contained HTML
"""

__version__ = "0.2.0"
ARCH_VERSION = "1.0"
