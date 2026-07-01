# visual-ml-model

**Read PyTorch model source code → a left-to-right architecture diagram** in the style of a
paper/README figure: multiple inputs on the left, the data spine through the middle, outputs
plus a pink loss column on the right, branching/merging, dashed loss & feedback edges, and an
**"ONLY DURING TRAINING"** band. The output is a single self-contained, dependency-free
`.html` file (inline SVG, "Save PNG" button) you can drop next to a paper or README.

The trick: **a large language model (Claude Code) reads the code** and decides the blocks,
the data flow, which branches are training-only, and where the losses are. Rule-based parsing
recovers the module list, but it can't tell you that a branch only runs under `if
self.training`, that several tensors are concatenated into one fusion block, or that a head's
output is compared against a target in a loss. That understanding is what the LLM supplies.

![AutoE2E architecture diagram](docs/autoe2e-arch.png)

> Autoware's **AutoE2E** (camera-only end-to-end driving), generated from the code: inputs
> (cameras, egomotion, visual history) on the left → backbone → view fusion → driving policy →
> outputs, with the self-supervised future-state branch + losses banded under "ONLY DURING
> TRAINING" on the right. Each box shows what it does + its tensor shape.

---

## Why

Existing PyTorch visualizers (Netron, torchview, torchviz, torchinfo, `torch.fx`, TensorBoard
`add_graph`, …) are op/shape/parameter dumps that need a *runnable* model and produce a flat
graph of low-level ops — not a readable "how does this system fit together" figure with
inputs, outputs, losses, and training-only paths. This tool instead has an LLM read the
**source** (no execution, no install needed) and emit a high-level architecture graph, then
lays it out automatically into a clean paper-style figure.

Research model code often isn't runnable, so the pipeline is **LLM-first**: it never imports
or runs the model. Models commonly use a Registry/factory pattern, so you pass the constructor
args as a JSON config — and you can **pick which registry variant** to draw.

---

## Install

```bash
# Python 3.11+. No third-party Python dependencies (validator + renderer are stdlib only).
# This tool ships inside the auto_e2e repo:
cd Tools/visual-ml-model

# The architecture-extraction step needs the `claude` CLI (Claude Code) on your PATH:
claude --version
# (Without it you can still render a pre-computed / hand-edited arch IR via `--arch`.)
```

---

## Usage

### Generate a diagram from source (live — Claude reads your model)

```bash
python -m viz_ml arch path/to/model.py \
    --class AutoE2E \
    --config examples/auto_e2e/autoe2e_bev.json \
    -o autoe2e.arch.html \
    --save-ir autoe2e.arch.json     # optional: also keep the generated IR
open autoe2e.arch.html
```

This resolves the target class + its same-repo submodules, hands the code + AST facts +
config to Claude, which emits an `arch_v1` IR; the IR is validated and rendered to a
self-contained HTML/SVG figure.

### Render a pre-computed / hand-edited IR (offline, no Claude call)

```bash
python -m viz_ml arch --arch examples/auto_e2e/autoe2e.arch.json -o autoe2e.arch.html
```

The IR is plain JSON — tweak a title, a shape, or an edge and re-render instantly.

### Registry / factory models — pick a variant

Many models build submodules through a registry/factory (e.g. AutoE2E selects its view-fusion
via `build_view_fusion(fusion_mode)` over `FUSION_REGISTRY = {"concat":…, "cross_attn":…,
"bev":…}`). The tool follows the factory to all candidates and lets you choose one via config:

```bash
python -m viz_ml variants path/to/auto_e2e.py --class AutoE2E
#   FUSION_REGISTRY:
#     "concat"      -> ConcatViewFusion
#     "cross_attn"  -> CrossAttentionViewFusion
#     "bev"         -> BEVViewFusion   ◀ ACTIVE  (when config says "fusion_mode":"bev")
```

The selected class is drawn as the active architecture; the others are excluded from the flow.

### Other commands

```bash
python -m viz_ml facts model.py --class AutoE2E      # Stage 0/1 AST extraction only (no LLM)
python -m viz_ml validate autoe2e.arch.json          # check an arch IR against the schema
```

Live extraction on AutoE2E (Claude-generated; inferred losses render faded with a `?`):

![AutoE2E arch, live extraction](docs/autoe2e-arch-live.png)

---

## The arch IR

A general left-to-right node+edge graph — see [`schema/arch_v1.schema.json`](schema/arch_v1.schema.json):

- **nodes**: `{id, title, desc, shape ("(8,256,8,8)"), role, train_only, ...}`. `role` drives
  the box color (input, backbone, fusion, policy, recurrent, head, output, future_state,
  prediction, learning_method, buffer, loss, …). `train_only` puts the box in the
  "ONLY DURING TRAINING" band.
- **edges**: `{from, to, kind}` where `kind ∈ {dataflow, loss, feedback, skip}` —
  dataflow = solid gray (used for left→right layering), loss = dashed pink into a loss node,
  feedback = dashed amber recurrent/conditioning loop, skip = thin gold shortcut.
- **groups**: the banner label ("ONLY DURING TRAINING").
- Every claim carries an optional `confidence`; inferred/low-confidence boxes render faded
  with a `?`, so the figure is honestly uncertain rather than confidently wrong.

The layout is **deterministic** (stdlib only): longest-path columns over dataflow edges,
barycenter row ordering, a separate lower band for training-only blocks, a dedicated rightmost
loss column, bezier forward edges, feedback edges routed through a reserved top channel, and a
banner drawn over the train-only members.

The Stage-3 instructions to Claude are in [`prompts/arch_system_prompt.md`](prompts/arch_system_prompt.md).

---

## Pipeline

```
source.py ─▶ [0] resolve ─▶ [1] AST facts ─▶ [3] Claude → arch_v1 IR ─▶ [4] validate + render ─▶ HTML
```

| Stage | What | LLM? | torch? |
|------|------|:---:|:---:|
| **0 resolve** | find the target class, follow same-repo imports, select the registry variant, assemble a code bundle | – | no |
| **1 AST facts** | stdlib `ast`: submodule inventory, `register_buffer(persistent=)`, forward() skeleton | – | no |
| **3 extract** | **Claude reads the bundle + facts + config and emits the arch_v1 IR** | **yes** | no |
| **4 validate + render** | stdlib schema + structural checks → deterministic layout → inline SVG → self-contained HTML | – | no |

Everything except Stage 3 is deterministic plumbing; the LLM does the one irreplaceable part
(understanding the code). Verified live on Autoware **AutoE2E** (multi-file, cross-file
relative imports, registry/factory view fusion) and a nanoGPT block.

## Repository layout

```
viz_ml/            resolve.py (stage 0) · ast_facts.py (1) · extract.py (3, Claude→arch IR)
                   validate.py · arch_render.py (layout + SVG) · cli.py
schema/            arch_v1.schema.json — the architecture IR contract
prompts/           arch_system_prompt.md — the Stage-3 instructions to Claude
examples/          auto_e2e/ (configs + a hand-authored & a live arch IR) · nanogpt/ (model + config)
docs/              architecture-diagram screenshots
```

## CLI commands

| Command | Purpose |
|---------|---------|
| `arch` | generate the left-to-right architecture diagram (the tool's one job). |
| `variants` | list the registry/factory variants the model can select among. |
| `facts` | Stage 0/1 AST extraction only (no LLM). |
| `validate` | check an arch IR against the schema + structural invariants. |
