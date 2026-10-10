# ruff: noqa
"""Instantiate the evaluated AutoE2E configuration and count parameters."""
from __future__ import annotations

import json
import sys
from collections import OrderedDict

sys.path.insert(0, "/tmp/autoe2e_src/Model")
sys.path.insert(0, "/tmp/autoe2e_src")

import torch  # noqa: E402

from training.reactive_multitask import (  # noqa: E402
    ReactiveTrainingStage,
    reactive_model_kwargs,
)

kwargs = reactive_model_kwargs(
    ReactiveTrainingStage.KITSCENES_FINETUNE, num_views=6
)
print(json.dumps(kwargs, indent=1, default=str))

from model_components.auto_e2e import AutoE2E  # noqa: E402

# Random init is fine for counting (no download of pretrained weights).
kwargs_local = dict(kwargs)
kwargs_local["is_pretrained"] = False
kwargs_local["backbone"] = "res_net_50"
model = AutoE2E(**kwargs_local)
reactive = model.Reactive_E2E


def count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


rows: "OrderedDict[str, int]" = OrderedDict()
rows["Backbone (ResNet-50 stages)"] = count(reactive.Backbone)
ff = reactive.FeatureFusion
rows["FeatureFusion.feature_pyramid (FPN)"] = count(ff.feature_pyramid)
vf = ff.view_fusion
rows["ViewFusion.bev_queries"] = vf.bev_queries.weight.numel()
rows["ViewFusion.row_embed+col_embed"] = (
    vf.row_embed.weight.numel() + vf.col_embed.weight.numel()
)
rows["ViewFusion.level_embeddings"] = vf.level_embeddings.numel()
rows["ViewFusion.camera_embeddings"] = vf.camera_embeddings.numel()
rows["ViewFusion.front_residual_gate"] = vf.front_residual_gate.numel()
rows["ViewFusion.pseudo_projection"] = vf.pseudo_projection.numel()
rows["ViewFusion.encoder layers (6)"] = count(vf.layers)
rows["  per encoder layer"] = count(vf.layers[0])
rows["    self-attention (T1 deformable)"] = count(vf.layers[0].self_attention)
rows["    spatial cross-attention"] = count(vf.layers[0].cross_attention)
rows["    FFN"] = count(vf.layers[0].ffn)
rows["    norms"] = count(vf.layers[0].norms)
rows["ViewFusion.front_cross_attention"] = count(vf.front_cross_attention)
rows["ViewFusion total"] = count(vf)
rows["TemporalFusion (T8 ResNet fusion)"] = count(ff.temporal_fusion)
rows["FeatureFusion total"] = count(ff)
rows["NavigationEncoder.MapEncoder (semantic raster)"] = count(
    reactive.NavigationEncoder.MapEncoder
)
rows["NavigationEncoder.RouteEncoder (lightweight)"] = count(
    reactive.NavigationEncoder.RouteEncoder
)
rows["NavigationEncoder.route_gate"] = reactive.NavigationEncoder.route_gate.numel()
rows["NavigationEncoder total"] = count(reactive.NavigationEncoder)
rows["MapBEVFusion (deformable)"] = count(reactive.MapBEVFusion)
rows["BEVSegmentationHead"] = count(reactive.BEVSegmentationHead)
rows["RouteReconstructionHead"] = count(reactive.RouteReconstructionHead)
rows["TemporalMemory (no_memory)"] = count(reactive.TemporalMemory)
tp = reactive.TrajectoryPlanner
rows["TrajectoryPlanner (GRU) total"] = count(tp)
rows["  planner.gru"] = count(tp.gru)
rows["  planner.ego_state_proj"] = count(tp.ego_state_proj)
rows["  planner.visual_history_proj"] = count(tp.visual_history_proj)
rows["  planner.value_proj"] = count(tp.value_proj)
rows["  planner.output_proj"] = count(tp.output_proj)
rows["  planner.sampling_offsets"] = count(tp.sampling_offsets)
rows["  planner.attention_weights"] = count(tp.attention_weights)
rows["  planner.reference_point"] = count(tp.reference_point)
rows["  planner.control_head"] = count(tp.control_head)
rows["  planner.ego_query"] = count(tp.ego_query)
rows["  planner.reasoning_coupling"] = count(tp.reasoning_coupling)
rows["ReactiveE2E total"] = count(reactive)
rows["AutoE2E total"] = count(model)

print("\n=== parameter counts (all) ===")
for k, v in rows.items():
    print(f"{k:55s} {v:>14,d}")

# Trainable set for the evaluated trajectory stages: frozen camera BEV.
reactive.freeze_camera_bev(adapt_temporal_running_stats=False)
# KITScenes finetune / nuPlan trajectory stage: bev head frozen (bev weight 0)
for p in reactive.BEVSegmentationHead.parameters():
    p.requires_grad_(False)
trainable = count(model, trainable_only=True)
frozen = count(model) - trainable
print("\n=== after freeze_camera_bev + frozen BEV head ===")
print(f"trainable: {trainable:,d}  frozen: {frozen:,d}  total: {count(model):,d}")
print("trainable modules:")
seen = set()
for name, p in model.named_parameters():
    if p.requires_grad:
        top = ".".join(name.split(".")[:3])
        if top not in seen:
            seen.add(top)
            print("  ", top)

# Sanity: planner num_points and fusion config
print("planner num_points:", tp.num_points, "offset_scale:", tp.offset_scale)
print("map fusion K:", reactive.MapBEVFusion.num_points, "heads:", reactive.MapBEVFusion.num_heads)
print("bev grid:", vf.bev_h, vf.bev_w, vf.pc_range)
print("nav encoder out:", reactive.NavigationEncoder.MapEncoder.output_h, reactive.NavigationEncoder.MapEncoder.output_w)
print("route hidden:", reactive.NavigationEncoder.RouteEncoder.stem[3].out_channels)
print("front gate trainable:", vf.front_residual_gate.requires_grad)
print("aux output size:", reactive.RouteReconstructionHead.output_size)
print("backbone feature channels:", reactive.Backbone.feature_channels)
