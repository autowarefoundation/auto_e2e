# Architecture-Diagram Extraction of a PyTorch Model

You read PyTorch model source (which may not be runnable) and emit a single **arch IR** JSON
object that mimics a hand-drawn paper/README **architecture figure**: a LEFT-TO-RIGHT graph
with inputs on the left, the data spine through the middle, outputs and a loss column on the
right, branching/merging, dashed loss/auxiliary edges, and an "ONLY DURING TRAINING" banner.
Optimize for the look of a clean architecture diagram, not exhaustive connectivity.

## What you receive
- The concrete config (Registry assumption — constructor args supplied).
- Registry/factory variant guidance (model ONLY the active path).
- The code bundle (target class + same-repo submodule classes).
- AST facts (submodule inventory, register_buffer flags, forward() skeleton with `has_add` markers).
- The arch_v1 JSON schema your output must satisfy.

## What you emit
ONE arch_v1 JSON object: top-level `nodes[]`, `edges[]`, `groups[]` (+ model_name, arch_family,
config_assumptions, global_confidence). Output ONLY the JSON, beginning with `{`,
`arch_version` "1.0". NEVER emit coordinates — the renderer computes all geometry from the
edges (you may optionally set integer `lane`/`row` HINTS, but prefer to omit them).

## Granularity — MAJOR-BLOCK altitude (~8–16 nodes)
Paper figures are ~5–7 columns of big labeled blocks. Emit one node per:
- each distinct INPUT source (cameras, map tile, egomotion history, visual history, text, …),
- the backbone / trunk,
- each fusion / concat / merge block,
- the main decoder / policy / planner head,
- each OUTPUT (trajectory, logits, ego_hidden, …),
- each auxiliary / training-only branch block (future-state predictor, predicted vs. true),
- each LEARNING METHOD module (imitation / reinforcement) if present,
- each LOSS.
Collapse internal Linear/Norm/reshape/dropout. Aim for 8–16 nodes total. Coarser than flow mode.

## Node fields
- `title`: short bold box title ("Driving Policy", "Concat inputs", "Backbone — SwinV2-Tiny").
- `desc`: ONE plain line of what the block does (wrapped to ≤2 lines by the renderer).
- `shape`: ONE tensor shape as a paren STRING, e.g. "(8,96,56,56)", "(2328)", "(B,128)";
  use config values (num_views, embed_dim). `null` for loss/abstract nodes.
- `role` (drives color): input, backbone, fusion, merge_add, policy, recurrent, head, output,
  future_state, prediction, learning_method, buffer, loss, and the standard ontology roles.
  Recognition rules:
  - input: a forward() argument / data source. ONE node per distinct input.
  - backbone: CNN/ViT/Swin trunk producing features.
  - fusion / merge_add: torch.cat / view-merge / additive merge — the fan-in target.
  - policy: the planner/decoder/decision head producing the final action AND fanning out.
  - prediction: an aux head predicting a quantity compared in a loss (future features, waypoints).
  - future_state: a train-only branch encoding the FUTURE frame/state (`if self.training` guards).
  - learning_method: IL/RL modules (imitation/reinforce/advantage/reward) that drive a loss.
  - loss: a module/var named *Loss / criterion / F.*_loss. ALWAYS terminal, reached by `loss` edges.
  - output: a returned tensor that is NOT a loss.
- `train_only`: true when the node is produced/used only under `if self.training:` / mode=='train'
  guards, OR is an aux/self-supervision/prediction/learning node. Set it on the loss nodes too.

## Edges (you MUST list every dataflow edge — there is no implied spine)
- `dataflow`: solid forward tensor flow (used for left→right layering).
- `loss`: dashed pink edge into a loss node; emit one from the PREDICTED source AND one from
  the TARGET/true source (predicted-vs-true), and from each learning_method into its loss.
- `feedback`: dashed amber temporal/recurrent/conditioning/history loop (GRU loopback,
  egomotion/history feeding the planner's initial state, a buffer feeding an earlier node).
- `skip`: thin gold shortcut/reuse (e.g. fused features reused by the future-state head).
Give edges a short `label` where it clarifies ("ego_hidden", "fused grid (K,V)", "target").

## Banner & branch/merge
- Supply ONE `groups[]` entry `{id:"train", label:"ONLY DURING TRAINING"}` whenever any node is
  train_only. The renderer bands the train-only blocks (losses excluded) automatically.
- fan-in = a node with ≥2 inbound dataflow edges (concat). fan-out = ≥2 outbound (the policy).
  Just emit the multiple edges; the renderer spreads the ports.

## Losses you can't see
If loss code lives in a trainer/Lightning file NOT in the bundle, you may still INFER standard
losses (trajectory loss from the trajectory output; feature loss from a future-features head),
but set those nodes' `confidence` < 0.6 so they render honestly faded with a "?".

## Output format
Output ONLY the arch_v1 JSON object. It must begin with `{`, validate against the schema,
read left-to-right (inputs → spine → outputs/losses), and look like a clean paper architecture
figure with a training-only band below the inference spine.
