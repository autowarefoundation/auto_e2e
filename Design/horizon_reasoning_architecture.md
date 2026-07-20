# Design Document: Horizon-Aware Action-Relevant Reasoning Branch

## Document Metadata

| Field | Value |
|-------|-------|
| Status | Proposed (supersedes the #108/#109 v1 skeleton) |
| Authors | riita10069 |
| Created | 2026-07-08 |
| Related Issue | [#98 — Proposal: Reasoning band (1 Hz scenario-description / VLM student–teacher)](https://github.com/autowarefoundation/auto_e2e/issues/98) |
| Builds on | #85/#93 (Encoded Visual History), #103 (per-horizon confidence), #107 (projection ABI), #108/#109 (reasoning v1 skeleton, teacher registry, evaluation) |
| Supersedes | The 3-axis taxonomy (`maneuver`/`edge_case`/`weather_env`) from #108 |

---

## 1. Executive Summary

This document specifies the **Reasoning branch** of AutoE2E: a lightweight, offline-supervised,
**horizon-aware, action-relevant** reasoning head that predicts *why* the planner should change
its behaviour over a short future horizon (now, +1s, +2s, +3s, +4s @ 1 Hz), and conditions the
trajectory planner through a **zero-initialised gate**.

The design rests on three positions the working group converged on in the #98 thread:

1. **Reasoning ≠ language generation.** The runtime student does not emit text. It projects the
   already-computed Encoded Visual History into a *finite action-relevant semantic space* (what
   matters to ego, when, why, which response class, how confident). Natural language is kept only
   as an optional offline audit trail (`evidence`), never on the runtime path.
   (riita, #98: *"Reasoning = predicting action-relevant meaning, causes, and future constraints
   from perception."*)

2. **The reason is the value, not the action.** The predicted trajectory already encodes the
   action (accelerate / stop / steer). What is *additive* is the reason behind it — the
   **action–reason pair** (`prepare_stop ← vru_conflict`, `stay_stopped ← red_light`).
   (m-zain-khawaja, #98: the action description is not value-additive since the trajectory encodes
   it; the reason is what is informative.)

3. **Teacher models are train-only and model-agnostic.** All teacher inference (Cosmos3-Nano,
   Qwen, VideoLLaMA, or a mock) happens **offline**, behind an OpenAI-compatible endpoint, and is
   consumed as **frozen, versioned label artifacts**. No teacher weight, client, or import ever
   enters `model_components/`, the training forward pass, or the vehicle. The default distributed
   artifact stays Apache-2.0; Cosmos is used only as an optional offline labeller (OpenMDW-1.1
   imposes no restriction on *outputs*, only on redistributing weights).

This branch is **opt-in and default-off**: with reasoning disabled the reactive/world-model
baseline is byte-identical. With it enabled but untrained, the zero-init gate makes the trajectory
byte-identical up to numerical tolerance — the branch only affects planning once training moves
the gate away from zero.

### Why this supersedes the v1 skeleton

PR #108/#109 shipped a correct **v1 skeleton** — per-horizon multi-label heads, a per-horizon
confidence head, a zero-init FiLM gate, Asymmetric Loss kept outside the model, an offline teacher
registry with an OpenAI-compatible endpoint, and a long-tail + faithfulness evaluation harness.
This document keeps those good bones and replaces the two weak parts that the authors themselves
flagged:

| v1 skeleton (#108/#109) | This design (v2) |
|---|---|
| Shared **MLP trunk** over the 896 vector | **Context tokens + 5 horizon queries + cross-attention decoder** → per-horizon `[B,5,256]` tokens |
| **3-axis taxonomy** (`maneuver`/`edge_case`/`weather_env`) | **Compositional action-relevant ontology** (relation / hazard / cause / longitudinal / lateral / tactical / rule + optional scene/topology/actor/timing) |
| Gate conditioned on **current-horizon** probs only; modulates the visual history | **`reasoning_latent [B,256]`** (AttentionPool over horizon tokens) as the single planner interface, plus `horizon_tokens` for horizon-aware cross-attention |
| Teacher clients live **inside** `model_components/reasoning/teachers/` | Teacher clients move to `Model/data_processing/reasoning_label_generation/`; `model_components` stays runtime-safe (no teacher import at all) |
| Targets are tensors returned directly by a teacher | **Versioned JSONL/Parquet artifacts** with provenance, confidence, and `abstained` records; training reads artifacts, never a live teacher |

Because the project is in a **research phase**, v2 does **not** preserve backward compatibility
with the v1 taxonomy or interfaces — the old 3-axis taxonomy and the `ReasoningBand` MLP are
removed, not wrapped. (See `CLAUDE.md` § "開発フェーズ方針".)

---

## 2. Motivation and Problem Statement

### 2.1 What the branch is for

From the 24/06 architecture and the #98 objective: **help the policy handle edge cases**, and let
a human introspect *what the model is reasoning about*. Average open-loop trajectory metrics are
dominated by ego status (BEV-Planner, [2312.03031](https://arxiv.org/abs/2312.03031)), so a branch
that improves long-tail behaviour must be measured on long-tail slices and by *intermediate concept
accuracy*, not only mean trajectory error.

### 2.2 Why front-camera-only 1 Hz is a prototype scope, not a claim

The v1 prototype consumes the front-biased Encoded Visual History at 1 Hz. Several edge cases —
cross traffic, side occlusions, cut-ins, VRUs entering from the side — are not reliably observable
this way. This design therefore:

- treats front-only as the *prototype* input,
- keeps a **front-only vs multi-camera reasoning** ablation as required follow-up,
- does **not** claim full long-tail coverage from front-only reasoning.

(riita, #98 point 4.)

### 2.3 Why not a generic frozen VLM at runtime

Zero-shot generic VLMs (Qwen3, Gemini, Moondream2) are unreliable on foundational ODD concepts such
as *intersection* and are expensive (~70 s/frame through the plain HF path, gcordova's datapoint;
Moondream2 zero-shot weak on road scenes, Zain's datapoint). The deployable path is therefore a
**small trained student head over the 896 vector**; generic VLMs are strictly offline teachers or
baselines. Structured labels are ~+20% decision accuracy and >10× faster than free text
([2506.05442](https://arxiv.org/pdf/2506.05442)); train-only VLM supervision is the recipe that
actually moves collisions (VLM-AD −38.7/−57.4%, [2412.14446](https://arxiv.org/abs/2412.14446);
Senna −33%, [2410.22313](https://arxiv.org/abs/2410.22313)).

### 2.4 Design requirements

| # | Requirement | Rationale |
|---|-------------|-----------|
| R1 | No online teacher inference in the forward pass / training loop | Reproducibility, cost, local testability, no external-availability coupling |
| R2 | Teacher endpoint is model-agnostic (`provider`/`base_url`/`model`/`prompt_version`/`schema_version`) | Cosmos / Qwen / mock behind one OpenAI-compatible boundary |
| R3 | Label generation lives in Flyte Processing, runnable locally without Kubernetes | Contributor-friendly (mock/cached), scalable (endpoint/EKS) |
| R4 | Labels are pre-extracted and versioned (JSONL + Parquet, full provenance) | No training from raw API responses; auditable, filterable |
| R5 | Labels are action-relevant and compositional | Answer what / when / why / how-respond / how-confident, not a flat enum |
| R6 | The head is horizon-aware — `horizon_tokens [B,5,256]` preserved, not only pooled | Timing of hazards must survive to the planner |
| R7 | Planner coupling is zero-init and optional (`none`/`pooled_latent`/`horizon_cross_attention`) | Byte-identical baseline at init; ablation-ready |
| R8 | Loss supervises structured labels **and** confidence | An exposed confidence output must be trained, not decorative |
| R9 | Endpoint failures are strict by default; abstentions are explicitly marked | Never silently poison the dataset with all-zero labels |
| R10 | Multi-teacher fusion validates exact taxonomy compatibility (labels + order) | Index i must mean the same class for every teacher |

---

## 3. System Overview

### 3.1 Two clocks, one interface

```
                 OFFLINE (preprocessing, train-only)          RUNTIME (vehicle-safe, no VLM)
 ┌────────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ dataset sample (clip + ego + route/map + logs)  │   │ Encoded Visual History [B,896]         │
 │        │                                         │   │  + ego_context [B,256]                 │
 │        ▼                                         │   │  + optional route/map context          │
 │ Flyte Processing task                            │   │        │                               │
 │  provider = mock | cached | openai_compatible    │   │        ▼                               │
 │        │                                         │   │ HorizonReasoningHead                   │
 │        ▼                                         │   │  context tokens → 5 horizon queries    │
 │ OpenAI-compatible teacher endpoint               │   │  → cross-attn decoder                  │
 │  (Cosmos3-Nano on vLLM / Qwen / mock)            │   │  → horizon_tokens [B,5,256]            │
 │        │                                         │   │  → structured logits + confidence      │
 │        ▼                                         │   │  → reasoning_latent [B,256]            │
 │ parse → validate schema → provenance             │   │        │                               │
 │        │                                         │   │        ▼                               │
 │        ▼                                         │   │ zero-init gate (α=0 at init)           │
 │ versioned artifacts (JSONL + Parquet)            │   │        │                               │
 │  reasoning_labels_v2.parquet                     │   │        ▼                               │
 └───────────────────────┬──────────────────────────┘   │ Trajectory Planner (Bezier / Flow)     │
                         │                                └──────────────────────────────────────┘
                         │ consumed as frozen targets
                         ▼
                 HorizonReasoningLoss  (in the training loop, outside the model)
```

The **only** coupling between offline and runtime is the frozen label artifact. The runtime student
never imports a teacher; the training loop reads Parquet, never calls an endpoint.

### 3.2 Where it plugs into the existing model

The existing pipeline is `AutoE2E → ReactiveE2E`. Inside `ReactiveE2E.forward` the relevant order
today is:

```
Backbone → FeatureFusion(BEV, projection ABI) → MapEncoder → MapBEVFusion
      → TemporalMemory(visual_history, egomotion_history) → (visual_ctx, ego_ctx)
      → TrajectoryPlanner(fused_features, visual_ctx, ego_ctx) → trajectory
```

The reasoning branch is inserted **after `TemporalMemory`**, because `ego_ctx` is exactly the
`ego_context [B,256]` the head needs and `visual_ctx` is the effective visual history the planner
sees (the World Model, when on, replaces the caller's raw `visual_history` with the WAM-aggregated
one before this point). The head consumes `(visual_ctx, ego_ctx)`, produces `reasoning_latent` /
`horizon_tokens`, and the planner consumes them behind the zero-init gate. This is the "integration
after context creation" the handoff prefers, and it makes horizon-token cross-attention natural (the
planner's action tokens attend to the 5 horizon tokens directly).

> Implementation note: today the branch is wired at the `AutoE2E` level (it modulates
> `visual_history` before `ReactiveE2E`). v2 moves the integration point **into `ReactiveE2E`**,
> after `TemporalMemory`, so the head sees `ego_ctx` and the planner coupling is a first-class
> planner argument rather than a pre-`ReactiveE2E` history rewrite. This is the one structural change
> to the model wiring.

---

## 4. Label Taxonomy (Compositional, Action-Relevant)

### 4.1 Principle

The label space is **not** a flat scene-fact enum. It is a set of independent axes ("groups"),
each of which answers a distinct question. Following the #98 convergence, the *action-facing* axes
(response / hazard / relation / cause) are the primary contract; scene/topology/actor axes are
supporting context with lower loss weight.

The taxonomy is the **single source of truth** in `Model/model_components/reasoning/reasoning_taxonomy.py`.
Two hard rules (inherited from the v1 `ScenarioTaxonomy`, which had this right):

- **Label order within a group is part of the loss contract.** Index `i` is a fixed class. Append
  only; never insert or reorder.
- **Every group includes an `unknown_*` label**, so a teacher can abstain within a group without
  producing an all-zero (falsely-negative) row.

`ReasoningTaxonomy` exposes `group_names()`, `labels(group)`, `num_classes(group)`,
`index(group, label)`, and `validate_exact_match(other)` (R10: same groups, same labels, same order,
same counts, same `schema_version`).

### 4.2 Minimum v1 groups (the action-relevant core)

These are required. Multi/single-label mode is fixed per group (part of the loss contract).

**`relation_to_ego`** (single-label; how the salient object relates to the ego path)
```
same_lane_ahead, same_lane_behind, left_adjacent, right_adjacent, crossing_path,
about_to_cross_path, merging_into_ego_path, cutting_into_ego_path, oncoming_conflict,
intersection_conflict, blocking_current_lane, blocking_target_lane, blocking_route,
occluded_near_path, outside_path, behind_ego, unknown_relation
```

**`hazard_event`** (multi-label; what risk is present)
```
no_hazard, collision_risk, vru_collision_risk, cut_in_risk, merge_conflict_risk,
right_of_way_violation_risk, red_light_violation_risk, blocked_route_risk, occlusion_risk,
low_friction_risk, emergency_vehicle_risk, unknown_hazard
```

**`cause`** (multi-label; *why* the planner may change behaviour — the hardest, most valuable head)
```
lead_vehicle, slow_lead_vehicle, stopped_lead_vehicle, cut_in_vehicle, cross_traffic,
oncoming_vehicle, pedestrian_crossing, pedestrian_about_to_cross, vru_conflict, red_light,
yellow_light, stop_sign, yield_sign, human_direction, route_turn, route_merge, route_lane_change,
lane_ending, object_blocking_path, blocked_lane, road_closed, construction_blocking_path,
occlusion, poor_visibility, slippery_road, uncertainty_high, unknown_cause
```

**`longitudinal_response`** (single-label)
```
keep_speed, accelerate, coast, slow_down, prepare_stop, stop, stay_stopped, creep, yield,
follow_lead_vehicle, increase_gap, emergency_brake, unknown_longitudinal
```

**`lateral_response`** (single-label)
```
keep_lane, nudge_left, nudge_right, shift_left_within_lane, shift_right_within_lane,
lane_change_left, lane_change_right, avoid_left, avoid_right, return_to_lane, pull_over, reverse,
unknown_lateral
```

**`tactical_response`** (single-label)
```
proceed, proceed_with_caution, wait, wait_for_gap, wait_for_actor, wait_for_signal,
creep_for_visibility, negotiate_merge, negotiate_unprotected_turn, yield_then_proceed,
stop_then_proceed, reroute_or_wait, unknown_tactical
```

**`rule_response`** (single-label)
```
none, wait_for_green, stop_at_stop_line, stop_before_crosswalk, yield_to_vru, yield_to_oncoming,
yield_to_cross_traffic, yield_to_emergency_vehicle, obey_human_direction, respect_speed_limit,
slow_for_school_zone, slow_for_construction_zone, do_not_enter, do_not_turn, unknown_rule
```

**`confidence`** — not a taxonomy group; a per-horizon scalar head (see §5.6).

### 4.3 Optional v2 groups (behind flags, default off)

`global_scene_context`, `ego_mission_context`, `road_topology`, `lane_topology`, `traffic_control`,
`right_of_way`, `dynamic_actor_type`, `actor_state`, `actor_intent`, `interaction_type` (all
multi-label context), plus continuous timing (`time_to_conflict`, `time_to_collision`,
`time_to_stop_line`) and the training-only `teacher_reasoning_embedding`. These append to the
taxonomy without touching v1 indices. v1 must not block on them.

### 4.4 Why split response into 4 axes

`slow_down` alone is ambiguous — it can be caused by a lead vehicle, a red light, a route turn, an
occlusion, or a comfort policy. Splitting longitudinal / lateral / tactical / rule makes the
supervision precise and directly useful to the planner. (riita, #98 §5.4.)

---

## 5. Runtime Student: `HorizonReasoningHead`

File: `Model/model_components/reasoning/horizon_reasoning_head.py`.
No teacher imports. `hidden_dim = 256` throughout.

### 5.1 Input contract

Required:
```
visual_history: [B, 896]     # Encoded Visual History (1 Hz slow path)
ego_context:    [B, 256]     # ego-motion context from TemporalMemory (ego_ctx)
```
Optional (omit the token entirely if absent — no learned null token):
```
route_context: [B, D_route]
map_context:   [B, D_map]
```

### 5.2 Context token projection

Each source is projected by its own MLP into a shared 256-d token:
```
MLP(x) = Linear(256→256) ∘ GELU ∘ Linear(D_in→256) ∘ LayerNorm(D_in)
```
```
visual_token = MLP_v(visual_history)   # [B,256]
ego_token    = MLP_e(ego_context)      # [B,256]
route_token  = MLP_r(route_context)    # [B,256]  (optional)
map_token    = MLP_m(map_context)      # [B,256]  (optional)

context_tokens = stack(available tokens)   # [B, N_context, 256];  v1 min: N_context = 2
```
Separate tokens keep the semantics of each source distinct rather than fusing too early.

### 5.3 Horizon queries + decoder

```
horizon_queries: nn.Parameter[5, 256]          # now, +1s, +2s, +3s, +4s
queries = horizon_queries.expand(B, 5, 256)

horizon_tokens = TransformerDecoder(           # cross-attn: query=queries, key=value=context_tokens
    num_layers=2, num_heads=4, dropout=0.1, activation=GELU
)(queries, context_tokens)                      # [B, 5, 256]
```
A pedestrian irrelevant *now* may be action-relevant in 2 s; each horizon needs its own
representation. A 2-layer / 4-head decoder is more expressive than one MLP and far cheaper than a
language decoder or a second vision pass — appropriate for a 1 Hz branch.

### 5.4 Structured heads

Each head maps `horizon_tokens [B,5,256] → [B,5,C]` via a per-group `Linear(256, C)`:
```
relation_to_ego_logits        [B,5,C_relation]     (single-label → CE)
hazard_event_logits           [B,5,C_hazard]       (multi-label → BCE/ASL)
cause_logits                  [B,5,C_cause]        (multi-label → BCE/ASL)
longitudinal_response_logits  [B,5,C_longitudinal] (single-label → CE)
lateral_response_logits       [B,5,C_lateral]      (single-label → CE)
tactical_response_logits      [B,5,C_tactical]     (single-label → CE)
rule_response_logits          [B,5,C_rule]         (single-label → CE)
confidence_logits             [B,5,1]
```
Optional v2 heads (§4.3) attach the same way behind flags.

### 5.5 Pooled `reasoning_latent`

```
reasoning_latent = MLP(AttentionPool(horizon_tokens))   # [B,256]
```
Always returned. This is the compact runtime interface the planner consumes. It must **not** be the
*only* interface — `horizon_tokens` is also exposed so the planner can attend to per-horizon timing
(R6). The structured heads shape the latent during training; the latent carries it to the planner.

### 5.6 Confidence head

Per-horizon `confidence_logits [B,5,1]`; `confidence = sigmoid(...)`. Trained (§7.4). Teacher labels
are noisy, so the model learns when its own reasoning is reliable — useful for gating and for the
drift/OOD monitor (§9.4). v1 is a single confidence head; splitting into
perception/prediction/rule/response confidence is v2.

### 5.7 Optional teacher-embedding alignment head (training-only)

```
student_reasoning_embedding = LayerNorm(Linear(256, D_teacher))(horizon_tokens)   # [B,5,D_teacher]
```
`D_teacher ∈ {512, 768}`. Disabled by default. When offline preprocessing produced a
`teacher_reasoning_embedding`, this head is aligned to it (§7.5). It captures soft semantics the
discrete heads drop. Per intisar's #98 question, it may **optionally** be kept live at inference as a
**read-only** signal for a student↔cached-teacher cosine drift / OOD check — but it is never required
by the planner, and feeding it to the planner is an ablation, not the default.

### 5.8 Output dataclass

`Model/model_components/reasoning/types.py`:
```python
@dataclass
class HorizonReasoningPrediction:
    horizon_tokens: torch.Tensor            # [B,5,256]
    reasoning_latent: torch.Tensor          # [B,256]

    relation_to_ego_logits: torch.Tensor
    hazard_event_logits: torch.Tensor
    cause_logits: torch.Tensor
    longitudinal_response_logits: torch.Tensor
    lateral_response_logits: torch.Tensor
    tactical_response_logits: torch.Tensor
    rule_response_logits: torch.Tensor
    confidence_logits: torch.Tensor         # [B,5,1]

    # optional v2 (None unless enabled)
    student_reasoning_embedding: torch.Tensor | None = None
    time_to_conflict: torch.Tensor | None = None
    # ... other optional heads
```
The planner requires only `reasoning_latent` (and, in `horizon_cross_attention` mode,
`horizon_tokens`). Everything else is for training, metrics, debugging, and visualisation.

---

## 6. Planner Coupling (Zero-Init, Optional)

`reasoning_mode ∈ {"none", "pooled_latent", "horizon_cross_attention"}`.

All coupling is behind a **zero-initialised scalar `alpha` (or a zero-init projection)**, so at
initialisation the trajectory is byte-identical to the reasoning-off baseline up to numerical
tolerance (R7). This mirrors the repo's existing `ResidualMapFusion` α=0 pattern and #108's
`ZeroInitGate`.

### 6.1 `none`
No reasoning branch runs. Baseline unchanged.

### 6.2 `pooled_latent`
```
planner_context = base_context + alpha * reason_proj(reasoning_latent)     # alpha = 0 at init
```
- **Bezier**: `base_context` is the planner's pre-decode context (the `h_0` built from
  `ego_state_proj + visual_history_proj` in `BezierPlanner`). Add the gated residual there.
- **Flow-matching**: `base_mod_cond` is the AdaLN global conditioning vector
  (`visual_history_proj + ego_state_proj`). Add the gated residual, then
  `gamma, beta = adaln_modulation(mod_cond + t_emb).chunk(2)`. Least-invasive integration point.

### 6.3 `horizon_cross_attention`
The planner attends to the **5 horizon tokens**, preserving *when* a hazard matters.
- **Bezier**:
  ```
  planner_query   = base_context.unsqueeze(1)                       # [B,1,256]
  reasoned        = CrossAttention(planner_query, horizon_tokens, horizon_tokens).squeeze(1)
  planner_context = base_context + alpha * reason_proj(reasoned)    # alpha = 0 at init
  ```
- **Flow-matching**: the action queries `[B,T,256]` (one per future timestep) cross-attend the
  horizon tokens, so timestep-t actions can look at the now/1s/2s/3s/4s reasoning:
  ```
  reasoned       = CrossAttention(action_queries, horizon_tokens, horizon_tokens)
  action_queries = action_queries + alpha * reason_proj(reasoned)  # alpha = 0 at init
  ```

### 6.4 Required ablation surface
The implementation must support A (none) / B (pooled_latent) / C (horizon_cross_attention) without
hard-coding `[B,256]` as the only path — this is exactly the ablation the faithfulness evaluation
needs (§9).

---

## 7. Loss Design

File: `Model/training/losses/horizon_reasoning_loss.py`. Computed in the training loop, **outside**
the model forward (same policy as the JEPA loss, #85/#13).

### 7.1 Total loss
```
L_total = L_planner
        + lambda_structured * L_structured
        + lambda_confidence * L_confidence
        + lambda_temporal   * L_temporal
        + lambda_alignment  * L_alignment
```
Defaults: `lambda_structured = 0.5`, `lambda_confidence = 0.05`, `lambda_temporal = 0.1`,
`lambda_alignment = 0.0` for v1 (`0.5` if teacher embeddings exist). If planner training
destabilises: `lambda_structured = 0.25`.

`L_planner` remains the primary signal; reasoning never replaces trajectory supervision.

### 7.2 Structured loss
Per head, averaged over the 5 horizons; summed over heads with per-head weights.
- **Multi-label** (`hazard_event`, `cause`, + optional context axes): `BCEWithLogitsLoss` default,
  `AsymmetricLoss` ([2009.14119](https://arxiv.org/abs/2009.14119)) option for the heavy
  negative-label imbalance.
- **Single-label** (`relation_to_ego`, the four response heads): `CrossEntropyLoss(ignore_index=-100)`.
- **Timing** (optional): `SmoothL1Loss`, computed only where a target exists.

Per-head weights (action-facing heads dominate):
```
longitudinal / lateral / tactical / rule / hazard : 1.0
relation_to_ego                                    : 0.8
time_to_conflict                                   : 0.8
cause / right_of_way / traffic_control / interaction : 0.5
actor_state / actor_intent                         : 0.4
scene / road_topology / lane_topology / actor_type : 0.2
```
`cause` is down-weighted to 0.5 because `weak_cause ≠ true cause` — the log cannot tell whether the
pedestrian, the red light, or the lead vehicle caused a slowdown (causal confusion). Its weight is
raised only where a factor is interventionally grounded (`counterfactual_gt`, §10).

### 7.3 Source & confidence weighting
Every label carries `label_source` and `label_confidence`:
```
weighted_loss = source_weight * label_confidence * raw_loss
source_weight: audited_gt 1.0 · direct_gt 0.9 · derived_gt 0.7 · teacher_gt 0.5 · weak_gt 0.3
               · counterfactual_gt (per §10) · abstained → masked out
```
Human-audited and direct dataset labels dominate noisy teacher labels.

### 7.4 Confidence loss (Brier)
```
L_confidence = mean( (sigmoid(confidence_logits) - target_confidence)^2 )
```
If teacher confidence is missing, derive `target_confidence` from provenance
(audited 1.0 / direct 0.9 / derived 0.7 / teacher = agreement-fraction / weak 0.3; abstained masked).
A confidence output that is exposed **must** be trained (R8) — never surface an untrained confidence
as a meaningful signal. The natural label-free target is the cross-teacher **agreement fraction**
(high disagreement → low confidence), not a single teacher's self-reported token probability (VLM
overconfidence is not fixed by prompting or scale, [2405.02917](https://arxiv.org/abs/2405.02917)).

### 7.5 Alignment loss (optional, training-only)
```
L_alignment = mean_h ( 1 - cos( normalize(student_emb[h]), normalize(teacher_emb[h]) ) )
```
Cosine is the stable default; KL over softened distributions is a v2 alternative.

### 7.6 Temporal consistency (weak regulariser)
Adjacent horizons should be coherent but must not be forced (real scenes change fast):
```
single-label heads: KL( softmax(logits_h) || softmax(logits_{h+1}) )
multi-label heads : | sigmoid(logits_h) - sigmoid(logits_{h+1}) |
weighted by       : confidence_h * confidence_{h+1}
lambda_temporal   : 0.05–0.1
```
Applied to the action-facing heads (`longitudinal/lateral/tactical` responses, `hazard_event`,
`relation_to_ego`).

### 7.7 Three-stage schedule
```
Stage 1: reasoning head only         L = L_structured + L_alignment + L_temporal + L_confidence
Stage 2: enable planner gate (α=0)   L = L_planner + auxiliary reasoning losses
Stage 3: long-tail fine-tuning       up-sample action-relevant edge cases
```
The head learns stable semantics first; the planner then learns to use the latent.

---

## 8. Offline Label Generation

### 8.1 Placement (runtime-safety boundary)

```
Model/data_processing/reasoning_label_generation/
    __init__.py
    schema.py           # ReasoningHorizonLabel, ReasoningLabelRecord, ReasoningTargetBatch
    teacher_client.py   # abstract client (was teachers/base.py)
    openai_compatible.py# OpenAI-compatible backend (was teachers/openai_endpoint.py)
    mock_teacher.py     # deterministic, offline (was teachers/deterministic.py)
    cached_teacher.py   # read labels from a prior artifact
    prompt_builder.py   # per_frame + clip_horizons prompts over the taxonomy
    validators.py       # 5-horizon / schema / taxonomy checks
    parquet_writer.py   # JSONL (debug) + Parquet (training)
    flyte_tasks.py      # local-runnable Flyte Processing tasks
```
`model_components/` imports **none** of this. The taxonomy
(`model_components/reasoning/reasoning_taxonomy.py`) is the shared contract both sides import; it has
no teacher dependency.

### 8.2 Teacher endpoint contract (R2)

Model-agnostic; depends only on `provider`, `base_url`, `model`, `prompt_version`, `schema_version`,
`request_mode`, `timeout`, `api_key`/secret-ref. Backends: `openai_compatible`, `mock`, `cached`,
`rule_based`. The network boundary is a single injectable `transport(url, payload, headers) → json`,
so CI runs with a stub (no network, no GPU).

Request modes:
- `per_frame` — one image → one horizon label. Simple backends / tests only.
- `clip_horizons` (preferred for real labelling) — current + future frames + ego/route/map context
  → one JSON with all 5 horizons. Prefer `video_url` if supported, else multiple `image_url`; prefer
  presigned HTTP or base64 data URLs (do not assume the endpoint reads `s3://`).

Failure policy (R9): `strict=True` default raises on endpoint/parse/schema error;
`strict=False` abstains and records `{"abstained": true, "teacher_error": "...", "provenance":
"teacher_error"}`. Never convert a failure into unmarked all-zero labels.

### 8.3 Cosmos3-Nano PoC as one backend

The `openai_compatible` provider plugs straight into the deployed Cosmos3-Nano vLLM PoC (private
repo `cosmos3-nano-vllm-eks-poc`; EKS `cosmos3-vllm-poc`, us-west-2, g6e/L40S; OpenAI-compatible
`/v1/chat/completions` reachable via `kubectl port-forward` at `localhost:8000/v1`, model
`cosmos3-nano`). Config example:
```yaml
teacher_labeling:
  provider: openai_compatible
  base_url: http://localhost:8000/v1     # or the presigned endpoint
  model: cosmos3-nano
  prompt_version: action_relevant_reasoning_v2
  schema_version: reasoning_label_v2
  request_mode: clip_horizons
```
Cosmos is used *only* to generate offline labels; its weights are never redistributed and never a
runtime dependency (OpenMDW-1.1 permits use of outputs; the Apache-2.0 artifact stays clean).

### 8.4 Artifacts (R4)

JSONL (debug) + Parquet (training). Each record carries `dataset_name`, `dataset_version`,
`sample_id`, `timestamp`, `schema_version`, `teacher_provider`, `teacher_model`,
`teacher_endpoint_type`, `prompt_version`, `request_mode`, `labeler_version`, `provenance`,
`created_at`, `abstained`, `teacher_error`, and exactly **5 horizons** in order. Validator fails on
missing/duplicated/unordered horizons or unknown labels.

Layout:
```
s3://.../reasoning_labels/dataset=<name>/split=<train|val|test>/
    schema_version=reasoning_label_v2/teacher=<name>/prompt_version=<ver>/labels.parquet
```

### 8.5 Multi-teacher fusion (R10)

`validate_exact_match` before fusion: same group names, same labels per group, **same order**, same
counts, same `schema_version`. Fused target = per-label agreement fraction (doubles as the confidence
target). Two-teacher agreement is +47–55% F1 over the best single
([2510.01126](https://arxiv.org/pdf/2510.01126)).

---

## 9. Evaluation

Reuses and extends the #109 harness (`Model/evaluation/`), which already uses the current
projection/geometry ABI (not the old `camera_params`).

### 9.1 Ablations
`baseline_no_reasoning` / `reasoning_pooled_latent` / `reasoning_horizon_cross_attention`. Report
overall and per long-tail slice.

### 9.2 Long-tail slices
`pedestrian_about_to_cross`, `vru_collision_risk`, `cut_in_vehicle`, `merge_conflict_risk`,
`occlusion_risk`, `unprotected_turn`, `blocked_lane`, `emergency_vehicle`, `human_direction`,
`ambiguous_right_of_way`.

### 9.3 Faithfulness intervention
Reasoning can be decorative rather than causal (VLADriveBench,
[2606.12706](https://arxiv.org/pdf/2606.12706)). Measure trajectory delta between reasoning-active
and reasoning-disabled, plus targeted interventions: zero all `horizon_tokens`, zero only the 1s /
2s token, zero hazard logits, zero response logits, shuffle horizon order. Metrics:
`trajectory_l2_delta`, `control_delta`, `collision_delta` (if available), hazard-slice delta.

**Critical:** the intervention must compare against the **effective** visual/ego context the planner
actually used (WAM-aggregated when the World Model is on), captured via a forward hook — not the raw
caller-provided history. The buffer must be snapshot/restored so the coupled and intervened runs see
the same history (else an untrained gate shows a spurious non-zero delta). (#109's `faithfulness.py`
already does this correctly and is carried forward.)

### 9.4 Intermediate concept accuracy
Average trajectory metrics can hide whether reasoning contributes. Also evaluate reasoning
*correctness*: intersection / crosswalk / object-in-path detection, VRU-risk, occlusion-risk,
traffic-rule context, per-horizon risk consistency. If the branch cannot reliably identify
foundational concepts, a trajectory improvement is suspect. (riita, #98 point 7.)

Optionally: student↔cached-teacher embedding cosine as a drift / OOD signal (intisar's read-only use).

### 9.5 Speed benchmark
`speed_benchmark` gets a reasoning column measuring the 1 Hz cost across
`none / pooled_latent / horizon_cross_attention`. It must pass `ReasoningHead` only supported kwargs
(no `backbone` kwarg — the v1 review bug); `reasoning` may remain a result label.

---

## 10. Research Track: Counterfactual `cause` Labels (proposal, not v1)

`cause` is the most planner-relevant and hardest-to-label head. Correlational labels (VLM captions,
trajectory logs) inherit imitation learning's causal-confusion failure mode. gcordova's proposal
(#98), building on riita's `counterfactual_gt` provenance slot: reuse the JEPA World Model (#85) and
the intervention operator, offline and per sample —

1. take the context tokens the head already ingests (including object tokens from a detector — where
   Zain's RF-DETR idea fits as a visual-grounding context token);
2. drop one candidate factor's token (e.g. the pedestrian) and re-run the planner;
3. keep the factor in `cause` **only if the plan changes beyond a threshold**.

A cyclist safely in the next lane → excluded; a pedestrian entering the path → `cause = vru_conflict`.
Output provenance = `counterfactual_gt`, letting §7 raise `cause_weight` only where the factor is
interventionally grounded. Bounded by World-Model fidelity and factor separability; validate against
object-aware pixel counterfactuals (OCTET, CVPR'23) on a small set. Deriving `cause` supervision by
intervention in a learned world model (vs Alpamayo-R1 CoC / OmniDrive VLM pseudo-labels) is
genuinely SOTA-plus. Kept disabled and clearly research-only.

---

## 11. Acceptance Criteria (Full Feature)

```
1.  AutoE2E default behaviour is byte-identical when reasoning is disabled.
2.  No teacher model/endpoint is imported by runtime model code (model_components/, forward pass).
3.  Reasoning labels are generated offline and stored as versioned artifacts (JSONL + Parquet).
4.  Flyte Processing runs locally with mock/cached labels without Kubernetes.
5.  An OpenAI-compatible endpoint can drive real teacher labelling (Cosmos3-Nano PoC plugs in).
6.  HorizonReasoningHead returns horizon_tokens [B,5,256], reasoning_latent [B,256],
    structured logits, confidence_logits [B,5,1].
7.  Planner supports none / pooled_latent / horizon_cross_attention.
8.  Planner coupling is zero-init and no-op at initialisation (tested).
9.  Confidence output is trained (never exposed as an untrained signal).
10. Endpoint failures are strict by default; abstentions explicitly marked.
11. MultiTeacher validates exact taxonomy label order.
12. Unit tests require no GPU, no network, no external model, no Kubernetes.
13. Speed benchmark works with reasoning enabled.
14. Faithfulness evaluation uses the current AutoE2E forward ABI (projection/geometry_type/
    image_transform) and compares against the effective planner inputs.
```

---

## 12. Implementation Plan

Single feature branch (no PR split — research phase). Order: schema/taxonomy → head → loss →
planner coupling → AutoE2E/ReactiveE2E integration → evaluation → teacher pipeline. Small,
frequently-committed steps (per `CLAUDE.md`). Tests are CPU-only and run on EC2 before push.

| Phase | Deliverable | Key files |
|-------|-------------|-----------|
| 1 | Compositional taxonomy + label schema + validators | `reasoning/reasoning_taxonomy.py`, `data_processing/reasoning_label_generation/{schema,validators}.py` |
| 2 | `HorizonReasoningHead` + output dataclass | `reasoning/horizon_reasoning_head.py`, `reasoning/types.py` |
| 3 | `HorizonReasoningLoss` (structured/confidence/temporal/alignment, source-weighted) | `training/losses/horizon_reasoning_loss.py` |
| 4 | Planner coupling (3 modes, zero-init) for Bezier + Flow-matching | `trajectory_planning/{base,bezier_planner,flow_matching_planner}.py` |
| 5 | Integrate after TemporalMemory in ReactiveE2E; aux_outputs return contract | `model_components/{reactive_e2e,auto_e2e}.py` |
| 6 | Teacher pipeline moved to data_processing; OpenAI-compatible + mock + cached + Flyte | `data_processing/reasoning_label_generation/*` |
| 7 | Evaluation: ablations, long-tail, interventions, concept accuracy, speed column | `evaluation/*`, `speed_benchmark/*` |

Removed (research phase, no compat shim): the 3-axis `scenario_taxonomy.py`, the MLP `ReasoningBand`,
and `teachers/` under `model_components`.

---

## 13. References

Grounding for each decision is in the #98 thread. Key papers:

- BEV-Planner / "Is Ego Status All You Need?" — [2312.03031](https://arxiv.org/abs/2312.03031)
- VLM-AD (train-only VLM supervision, −38.7/−57.4% collision) — [2412.14446](https://arxiv.org/abs/2412.14446)
- Senna (meta-actions → planner, −33% collision) — [2410.22313](https://arxiv.org/abs/2410.22313)
- Structured labels vs free text (+~20% decision acc, >10×) — [2506.05442](https://arxiv.org/pdf/2506.05442)
- Asymmetric Loss — [2009.14119](https://arxiv.org/abs/2009.14119)
- Two-teacher agreement (+47–55% F1) — [2510.01126](https://arxiv.org/pdf/2510.01126)
- VLM overconfidence not fixed by scale/prompting — [2405.02917](https://arxiv.org/abs/2405.02917)
- VLADriveBench (reasoning can be decorative) — [2606.12706](https://arxiv.org/pdf/2606.12706)
- SimLingo (vision-only, language–action alignment) — [2503.09594](https://arxiv.org/abs/2503.09594)
- Alpamayo-R1 Chain-of-Causation (research-only weights) — [2511.00088](https://arxiv.org/abs/2511.00088)
- OCTET (object-aware counterfactual explanations) — CVPR 2023
- Cosmos-Embed1-448p (AV/Physical-AI video-text embedding) — https://huggingface.co/nvidia/Cosmos-Embed1-448p
- OpenMDW-1.1 (outputs unrestricted; weights not Apache-2.0) — https://openmdw.ai/license/1-1/
