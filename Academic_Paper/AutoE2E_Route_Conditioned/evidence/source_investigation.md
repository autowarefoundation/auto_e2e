# AutoE2E source investigation notes (branch experimental/reactive-bev-learning @ addebfa4, 2026-09-14)
Worktree: /tmp/autoe2e_src  (read-only). Paper workdir: /tmp/autoe2e_paper

## Locked model config (Model/training/reactive_multitask.py::reactive_model_kwargs)
- REACTIVE_MODEL_ARCHITECTURE_VERSION = "bevformer_v2_t8_split_navigation_v5"
- backbone res_net_50 (timm resnet50 features_only, 5 stages ch [64,256,512,1024,2048]); input_profile bevformer_v2 (BGR mean subtraction, BN frozen)
- view_fusion: architecture bevformer_v2_t8, bev_h=300, bev_w=200, pc_range=(-60,-60,-5,120,60,3) -> 0.6 m cells
  image_size=512, front_camera_index=0, front_image_size=1024, num_heads=8, num_levels=4, num_points=8 (SCA), num_encoder_layers=6,
  feedforward_channels=512, query_chunk_size=4096, activation_checkpointing=True; self-attn (T1 deformable) num_points=4, num_bev_queue=2 (current BEV duplicated)
  num_points_in_pillar=4 (z from -5 to 3 m); learned BEV queries 300*200*256; row/col embed 256/2 each; level_embeddings 4x256; camera_embeddings Vx256
  front branch: copy of final layer's cross-attention, content_delta_only, residual through tanh(front_residual_gate) zero-init; gate stays trainable under freeze
- FPN: BEVFormerFeaturePyramid uses last 3 backbone stages (512,1024,2048) -> 4 levels x 256 (3 lateral + extra stride-2)
- T8 temporal fusion: concat 8 BEVs (2048 ch) -> 3 BasicBlocks 512 ch (3x3 conv) -> Linear 512->256 + LayerNorm. Frame order (-7..0), history BEVs detached, computed no_grad in eval mode.
  frame interval 500,000 us; 7 history frames; StatefulCameraFPNCache for inference streams.
- map_type semantic_raster (SemanticRasterEncoder: 5x5 conv 14->96, GN, SiLU; depthwise stride-2 ->192, ->192 (stride 4); 1x1 to 256; local 1x1 96->256; sum; GN+SiLU) output 300x200 (bilinear resize of input raster)
- route encoder LightweightRouteEncoder hidden=96 (stem 3x3 2->48->96, context depthwise stride 2 + 1x1, upsample, 1x1 ->256, GN, SiLU) output 300x200
- SplitNavigationEncoder: nav = E_map(M) + sigmoid(route_gate[256 zeros init]) * E_route(R); returns route_contribution for recon head
- validity gates: map_valid default True, route_valid default False; enable_route_conditioning=False forces route_valid zeros
- map_fusion deformable: MapDeformableCrossAttentionFusion K=8 sample pts, 8 heads, 256 ch, pre-LN query, offset_proj Linear(256->16), attn_proj Linear(256->64), out_proj zero-init, FFN 256->512->256 (GELU, dropout .1) last linear zero-init; grid_sample bilinear border; residual
- temporal_memory no_memory (parameter-free passthrough; visual_history (B,896) zeros for KITScenes, egomotion (B,256) flattened 64x4)
- planner gru: GRUPlanner num_points=16, offset_scale=0.1; context = Linear(256->256)(ego) + Linear(896->256)(vis); ego_query Embedding(1,256); per step: query=hidden.detach()+ego_query -> reference_point sigmoid (2), sampling_offsets (16x2), attention_weights (16) softmax; grid_sample over value_proj(BEV) 300x200; output_proj; GRU(256,256); control_head Linear(256->2). 64 steps.
- aux: BEVSegmentationHead (8 classes, 64 hidden, 2 residual blocks; output 450x300) bev weight 0.0 in trajectory stages; RouteReconstructionHead (2 ch, output 450x300) from route_contribution; weight 1.0
- enable_world_model False, enable_reasoning False
- Loss: TrajectoryXYImitationLoss = SmoothL1 (beta 1 m) on integrated XY vs GT XY, masked by validity, /(2*valid); RouteReconstructionLoss = 0.5 BCE(pos_weight corridor_pos_weight) + 0.5 soft Dice for corridor; 0.25 * CenterNet-style focal heatmap for destination
- Rollout (training/losses/control_rollout.py, "semi_implicit_unicycle_v1"): v_t = max(v_{t-1}+a_t dt,0); theta_t = cumsum(v_t*kappa_t*dt); x=cumsum(v cos theta dt), y=cumsum(v sin theta dt); dt 0.1; float32
- Egomotion: 64 x [speed, acceleration, yaw_rate, curvature] at 10 Hz (6.4 s), derived by finite differences from poses; normalize speed/33, accel/8 (auto_e2e.normalize_egomotion); curvature = yaw_rate/max(speed,0.1)
- KITScenes benchmark protocol: 40 history steps (4.0 s real, left zero padded to 64), 50 future steps (5.0 s), "paper_protocol_approximation" -> 78.125% coverage
- Cameras (KITScenes): camera_base_front_center (long-range front), camera_ring_front_left, front_right, rear_left, rear, rear_right -> slots front, front_left, front_right, rear_left, rear, rear_right. camera_ring_front omitted (duplicates front coverage); stereo pair excluded.

## Parameter counts (instantiated, six views) total 79,906,522 (repo BENCHMARKS.md 8-cam: 79,907,034)
Backbone 23,508,032; FPN 3,278,592; bev_queries 15,360,000; row+col embed 64,000; level emb 1,024; cam emb 1,536; front gate 256; pseudo proj 12;
encoder 6 layers 4,940,928 (per layer 823,488: self-attn 230,080, SCA 328,960, FFN 262,912, norms 1,536); front_cross_attention 328,960; ViewFusion total 20,696,716
T8 temporal fusion 30,809,856; FeatureFusion total 54,785,164
NavigationEncoder 245,440 (map 167,200; route 77,984; gate 256); MapBEVFusion 350,288; BEVSegHead 165,000; RouteReconHead 17,218; GRU planner 835,380 (gru 394,752; vis proj 229,632; ego proj 65,792; value 65,792; out 65,792; offsets 8,224; attn 4,112; ref 514; ctrl 514; query 256)
Trainable in trajectory stages (frozen camera BEV, frozen BEV head): 1,448,582 ; frozen 78,457,940

## Navigation raster geometry (Model/navigation/geometry.py)
- AUTOE2E_NAVIGATION_GEOMETRY "autoe2e-bev-450x300-0p4m-v1": 450x300 @0.4 m, X -60..120, Y -60..60, ego anchor row 299.5 col 149.5; corridor 3.5 m; dest radius 2.0 m; rear clip 10 m  -> model-facing rasters for T8 reactive shards (reactive_data.py requires this contract; kit_scenes/dataset.py rasterizes natively with it since e0d55e79 2026-09-10)
- DEFAULT_NAVIGATION_GEOMETRY "kitscenes-v3-bev-1m-v1": 256x256 @1.0 m, X -85.5..170.5, Y -128..128, anchor row 170 col 127.5 (audit: 1 m/px covers 99.79% of 6.4 s endpoints) -> KITScenes v3 published navigation artifacts, legacy train_il, benchmark/navigation_metrics default, BEV-seg label source (nearest resample to 450x300)
- SemanticRasterEncoder bilinearly resizes input to 300x200 (same extent as camera BEV when geometry is AUTOE2E)
- 14 map channels: 0 drivable_area (lanelet polygons), 1 lane_boundary (left/right bounds, 1.0 m wide lines), 2 lane_centerline (1.0 m lines), 3 intersection (lanelet polygon heuristic via turn_direction), 4 crosswalk (subtype crosswalk polygons), 5 stop_line (1.0 m lines), 6 static_traffic_signal (points radius 1.0 m i.e. width 2.0), 7 traffic_direction_sin ((sin+1)/2), 8 traffic_direction_cos ((cos+1)/2), 9 traffic_direction_valid (directed lane centerline fields, width = 3.5 m), 10 known_map_area (map bounds polygon), 11 road_level ((clamp(level,-8,8)+8)/16), 12 road_level_valid, 13 overlapping_level_ambiguity (set when conflicting levels overwrite or primitive level != active route level). All map values in [0,1] float32; route mask uint8 binary.
- Route: channel 0 selected corridor = lane centerline sequence drawn 3.5 m wide (rear clip 10 m behind ego); channel 1 destination disk radius 2.0 m
- Route provenance (kit_scenes/navigation.py + lanelet2_matcher.py): route = Lanelet2 trace match of the full scene ego trace (HMM-style: candidate radius 8 m, dist sigma 2 m, heading sigma 20 deg; costs same 0, following .25, adjacent 1, disconnected 25) -> lane sequence the driver traversed; destination = scene end position ("kitscenes_scene_end", estimated_destination=True). Rasters rendered at 500 ms anchors (latest non-future pose) then SE(2)-warped to sample pose. "Leak-resistant": no future ego points rasterized, but route is a posteriori lane-level.
- Lanelet2 adapter: drivable = lanelet polygons (non-crosswalk); intersection = lanelets with turn_direction attribute heuristic; crosswalk = subtype crosswalk; stop lines from scene_map.get_stop_lines(); traffic signals from regulatory elements traffic_light; direction fields from lanelet centerlines.

## Evaluation
- evaluate_reactive_multitask horizons: 1s=10,2s=20,3s=30,5s=50,6p4s=64 steps; ADE_h = mean over samples of per-sample mean Euclidean error over valid steps<=h; FDE_h = error at step h over samples where valid; mean abs longitudinal = mean|dx| over valid steps; lateral = mean|dy|; nonfinite prediction = sample with any nonfinite control (replaced by zeros); coverage = valid/total timesteps.
- Open-loop metrics (evaluation/reactive_open_loop.py, "reactive_open_loop_metrics_v1"): footprint 4.8 x 2.0 m, 4 corners; drivable compliance rate = compliant steps / total steps (map_valid samples); success = all steps compliant; route corridor compliance/success same with 3.5 m corridor mask (>=0.5); route progress proxy = arc-length projection of predicted terminal point onto GT trajectory polyline; ratio = progress / GT length (supported if GT length > 1 m); comfort thresholds (nuPlan): lon accel [-4.05, 2.40], lat accel 4.89 (v^2 kappa), yaw rate 0.95 (v kappa), yaw accel 1.93, lon jerk 4.13, magnitude jerk 8.37; violation if any; comfortable rate = 1 - violation. Default geometry AUTOE2E 450x300.
- KITScenes evaluation: batch bf16 autocast ("cuda_bfloat16_autocast_v1"); Val primary route usage policy "route_zero_reused_image_bev_no_grad_v2" (counterfactuals enabled, reuse precomputed image BEV, no input gradient); Test "not_applicable_camera_only_v1"; input_track camera_map_route vs camera_only_missing_map_route; official val 117 scenes (sha 421858c6...), official test 206 scenes (sha 0b96c983...)
- Legacy eval gate in workflows.py:7360: passed = avg_ade < 2.0 and avg_fde < 4.0 (6.4 s internal). Cannot confirm this is the registered eval_gate_pass definition.
- Stage lineage: NUPLAN_FULL multitask (trajectory+route, frozen BEVFormer) may take a BEV_ONLY Stage A parent (camera BEV unfrozen, BEV seg loss) ; KITScenes finetune requires frozen Epoch-5 nuPlan parent profile "nuplan_trajectory_route_v1"; bev_weight must be 0; freeze_bevformer True.
- Official BEVFormer V2 R50 T8 checkpoint sha256 5585bc4d...; source repo fundamentalvision/BEVFormer @ 66b65f3a; bev queries and row/col position embeddings resized; camera embeddings adapted to view count; detector heads omitted; weight license NOASSERTION, training data CC-BY-NC-SA-4.0 (nuScenes)
- KITScenes source: HF KIT-MRT/KITScenes-Multimodal, data revision 6fde0034..., SDK revision 7765cdec...
- BENCHMARKS.md: T8 training memory (unfrozen diag, 8 cams, L40S BF16 batch 1): fwd 8.2008 s, bwd 3.9579 s, 21.33 GiB peak alloc; 79,907,034 params. Not production throughput.
