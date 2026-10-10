# ruff: noqa
"""Schematic figures 2, 3, 6, 7, 8, 14, 15."""
import sys
sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
from svgkit import SVG, MONO

OUT = "/tmp/autoe2e_paper/vivlio/figures/"

# ======================= Fig 2: temporal contract =======================
s = SVG(760, 310, font_size=10.5)
x0, x1 = 200, 730           # time axis from -6.4 s to +6.4 s
def tx(t):  # t in seconds
    return x0 + (t + 6.4) / 12.8 * (x1 - x0)
axis_y = 262
s.line(x0, axis_y, x1, axis_y, color="#333", width=1.2)
for t in range(-6, 7):
    s.line(tx(t), axis_y - 4, tx(t), axis_y + 4, color="#333")
    s.text(tx(t), axis_y + 16, f"{t:+d}" if t else "0", size=9, anchor="middle")
s.text((x0 + x1) / 2, axis_y + 32, "time relative to the current frame [s]  (10 Hz base rate = 0.1 s per step)", size=9.5, anchor="middle")
s.line(tx(0), 30, tx(0), axis_y, color="#d62728", dash="4,3", width=1.2)
s.text(tx(0) + 4, 40, "now (t = 0)", size=9, color="#d62728")

def band(y, t_a, t_b, label, fill, stroke, h=26, text_color="#222", note=None):
    s.add(f'<rect x="{tx(t_a)}" y="{y}" width="{tx(t_b) - tx(t_a)}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.1" rx="4"/>')
    s.text((tx(t_a) + tx(t_b)) / 2, y + h / 2 + 4, label, size=9.5, anchor="middle", color=text_color, weight="bold")
    if note:
        s.text(tx(t_b) + 6, y + h / 2 + 4, note, size=8.5, color="#444")

# camera frames: 8 frames at -3.5..0 step 0.5
y = 52
s.text(14, y + 17, "Camera (T8)", size=10, weight="bold")
for i, t in enumerate([-3.5 + 0.5 * k for k in range(8)]):
    col = "#3F6FAE" if t < 0 else "#d62728"
    s.add(f'<rect x="{tx(t) - 6}" y="{y}" width="12" height="26" fill="{"#E3EEF9" if t < 0 else "#FBE5E3"}" stroke="{col}" stroke-width="1.2" rx="2"/>')
    s.text(tx(t), y + 40, f"{t:+.1f}" if t else "0", size=7.5, anchor="middle", color=col)
s.text(tx(-3.5) - 8, y + 12, "7 history frames", size=8.5, anchor="end", color="#3F6FAE")
s.text(tx(-3.5) - 8, y + 23, "6 views @ 512 px, detached", size=8.5, anchor="end", color="#3F6FAE")
s.text(tx(0) + 10, y + 12, "current: 6 views @ 512 px", size=8.5, color="#d62728")
s.text(tx(0) + 10, y + 23, "+ front view @ 1024 px (additive)", size=8.5, color="#d62728")
s.text(tx(0) + 10, y + 34, "-> 3.5 s visual span at 2 Hz", size=8.5, color="#333")

# egomotion history 6.4 s at 10 Hz
y = 110
s.text(14, y + 17, "Egomotion", size=10, weight="bold")
band(y, -6.4, -4.0, "zero pad (benchmark)", "#F4F4F4", "#9A9A9A", text_color="#666")
band(y, -4.0, 0.0, "4.0 s real history (40 steps)", "#E3EEF9", "#3F6FAE")
s.text(tx(-6.4), y - 6, "64 steps x [speed, accel, yaw rate, curvature] = 256-d; training uses all 6.4 s", size=8.5)

# planner output 6.4 s
y = 168
s.text(14, y + 17, "Planner output", size=10, weight="bold")
band(y, 0.0, 6.4, "64 x (accel., curvature) @ 10 Hz -> rollout -> 64 XY", "#E2F3E4", "#2E8B57")

# supervision / evaluation
y = 214
s.text(14, y + 17, "Targets (benchmark)", size=10, weight="bold")
band(y, 0.0, 5.0, "50 valid steps: ADE/FDE @ 1,2,3,5 s; |lat|,|lon|", "#FCF4DB", "#B8860B")
band(y, 5.0, 6.4, "no target", "#F4F4F4", "#9A9A9A", text_color="#666")
s.text(tx(6.4) - 2, y - 6, "valid horizon coverage = 50/64 = 78.125 %", size=8.5, anchor="end", color="#B8860B")
s.text(14, 22, "Temporal contract (KITScenes benchmark protocol: 40 history / 50 future steps; training: 64 / 64)",
       size=10.5, weight="bold")
s.save(OUT + "fig02_temporal_contract.svg")

# ======================= Fig 3: encoder layer detail =======================
s = SVG(760, 480, font_size=10.5)
s.text(14, 22, "One BEVFormer V2 encoder layer (x6, frozen) and the additive front-camera residual branch", size=11, weight="bold")
bx, bw = 30, 250
s.box(bx, 40, bw, 46, "BEV queries Q  (60,000 x 256)", ["300 x 200 learned embeddings + row/col position (128+128)"], style="frozen")
s.box(bx, 104, bw, 72, "Temporal self-attention (T1 form)",
      ["deformable; queue = {Q, Q} (current BEV duplicated)", "8 heads x 4 points per queue, offsets from [Q, Q+pos]",
       "value_proj / output_proj 256 x 256; 0.23 M"], style="frozen")
s.box(bx, 194, bw, 96, "Multi-scale spatial cross-attention",
      ["reference points: 4 pillar heights z in [-5, 3] m", "projected by calibrated pinhole/fisheye operator",
       "8 heads x 4 FPN levels x 8 points per view", "visible-view average; + level & camera embeddings", "0.33 M"],
      style="frozen")
s.box(bx, 308, bw, 46, "FFN 256 -> 512 -> 256 (ReLU, dropout 0.1)", ["0.26 M; 3 x LayerNorm (post-norm)"], style="frozen")
s.box(bx, 370, bw, 36, "layer output -> next layer / current BEV", [], style="plain")
for ya, yb in ((86, 104), (176, 194), (290, 308), (354, 370)):
    s.arrow(bx + bw / 2, ya, bx + bw / 2, yb)
s.text(bx + bw + 8, 140, "LN", size=9); s.text(bx + bw + 8, 246, "LN", size=9); s.text(bx + bw + 8, 332, "LN", size=9)

# camera features side
s.box(310, 104, 170, 72, "Camera FPN features",
      ["6 views x 4 levels x 256 ch", "512 px base tiles", "history frames re-used (cached)"], style="input")
s.arrow(310, 240, bx + bw, 240, label="values", label_dy=-4)
s.line(395, 176, 395, 240, color="#333"); s.line(395, 240, 310, 240, color="#333")

# front residual branch
fx, fw = 500, 246
s.box(fx, 40, fw, 58, "Native front tile 1024 x 1024 (t = 0)",
      ["same ResNet-50 + FPN", "4 levels, one view"], style="input")
s.box(fx, 116, fw, 96, "Front cross-attention (frozen copy of layer 6)",
      ["reference points via the front-camera operator", "content-only delta: no level/camera embedding bias",
       "delta = LN6(Q + attn) - LN6(Q)", "masked to cells observed by the front camera"], style="frozen")
s.box(fx, 230, fw, 52, "Gate  tanh(g)  (g in R^256, zero-init)",
      ["only trainable camera-side parameter (256)"], style="trainable", badge="gate train")
s.box(fx, 300, fw, 52, "Current BEV  B_0 = B_enc + tanh(g) * delta",
      ["at init: B_0 == B_enc exactly (no-op)"], style="plain")
s.arrow(fx + fw / 2, 98, fx + fw / 2, 116); s.arrow(fx + fw / 2, 212, fx + fw / 2, 230); s.arrow(fx + fw / 2, 282, fx + fw / 2, 300)
s.arrow(bx + bw, 388, fx + 40, 388, label=None); s.line(fx + 40, 388, fx + 40, 352, color="#333")
s.text(fx + 46, 384, "B_enc (layer-6 output, current frame)", size=8.5)
s.box(fx, 412, fw, 40, "-> T8 temporal fusion (+ 7 detached history BEVs)", [], style="frozen")
s.arrow(fx + fw / 2, 352, fx + fw / 2 + 60, 412, elbow=fx + fw / 2 + 60)
s.legend(30, 470, [("input", "input"), ("frozen", "frozen"), ("trainable", "trainable")])
s.save(OUT + "fig03_encoder_layer.svg")

# ======================= Fig 6: navigation encoders + deformable fusion =======================
s = SVG(760, 470, font_size=10.5)
s.text(14, 22, "Split navigation encoding and deformable navigation fusion (all trainable)", size=11, weight="bold")
# E_map stack
mx, mw = 20, 230
s.box(mx, 40, mw, 40, "M: 14 x 450 x 300", ["bilinear resample -> 14 x 300 x 200"], style="input")
s.box(mx, 92, mw, 62, "E_map stem", ["Conv 5x5, 14 -> 96, GroupNorm(16), SiLU", "local path: Conv 1x1, 96 -> 256"], style="trainable")
s.box(mx, 166, mw, 86, "E_map context", ["depthwise 3x3 s2 + Conv 1x1 96 -> 192, GN, SiLU",
      "depthwise 3x3 s2 + Conv 1x1 192 -> 192, GN, SiLU", "Conv 1x1 192 -> 256, GN; bilinear x4 upsample"], style="trainable")
s.box(mx, 264, mw, 40, "E_map(M) = GN+SiLU(local + context)", ["256 x 300 x 200; 0.167 M params"], style="trainable")
for ya, yb in ((80, 92), (154, 166), (252, 264)):
    s.arrow(mx + mw / 2, ya, mx + mw / 2, yb)
# E_route stack
rx, rw = 270, 230
s.box(rx, 40, rw, 40, "R: 2 x 450 x 300 (binary)", ["bilinear resample -> 2 x 300 x 200"], style="input")
s.box(rx, 92, rw, 62, "E_route stem", ["Conv 3x3, 2 -> 48, GN, SiLU", "Conv 3x3, 48 -> 96, GN, SiLU"], style="trainable")
s.box(rx, 166, rw, 86, "E_route context", ["depthwise 3x3 s2 (96) + Conv 1x1 96 -> 96, GN, SiLU",
      "bilinear x2 upsample; sum with stem output", "Conv 1x1 96 -> 256, GN, SiLU"], style="trainable")
s.box(rx, 264, rw, 40, "E_route(R)", ["256 x 300 x 200; 0.078 M params"], style="trainable")
for ya, yb in ((80, 92), (154, 166), (252, 264)):
    s.arrow(rx + rw / 2, ya, rx + rw / 2, yb)
# gate and sum
s.box(rx, 318, rw, 40, "x sigmoid(g)   g in R^256, zero-init -> 0.5", ["route_contribution (-> reconstruction head)"], style="trainable")
s.arrow(rx + rw / 2, 304, rx + rw / 2, 318)
sx, sy = 145, 340
s.add(f'<circle cx="{sx}" cy="{sy}" r="12" fill="white" stroke="#333" stroke-width="1.3"/>')
s.text(sx, sy + 4.5, "+", size=15, anchor="middle", weight="bold")
s.arrow(mx + mw / 2, 304, sx, sy - 12)
s.arrow(rx, 338, sx + 12, 340)
s.text(sx, 374, "B_nav: 256 x 300 x 200", size=10, anchor="middle", weight="bold", family=MONO)
s.text(sx, 388, "validity gates: M *= map_valid, R *= route_valid (sample level)", size=8.5, anchor="middle")
# deformable fusion
fx, fw = 528, 216
s.box(fx, 40, fw, 46, "B_img (query)  256 x 300 x 200", ["frozen camera path output"], style="frozen")
s.box(fx, 100, fw, 174, "Deformable cross-attention",
      ["q = LayerNorm(B_img cell)", "offset_proj: Linear 256 -> 2K (K = 8 points)", "attn_proj: Linear 256 -> 8 heads x K",
       "sample B_nav bilinearly at cell + offsets", "(grid_sample, border padding)", "per-head softmax over K, weighted sum",
       "out_proj: Linear 256 -> 256 (zero-init)", "residual add; then FFN 256->512->256", "(GELU, dropout 0.1, last Linear zero-init)"],
      style="trainable", line_size=8.8)
s.box(fx, 290, fw, 46, "B_fused = B_img + F(B_img, B_nav)", ["0.350 M params; exact identity at init"], style="trainable",
      mono_lines=("B_fused = B_img + F(B_img, B_nav)",))
s.arrow(fx + fw / 2, 86, fx + fw / 2, 100); s.arrow(fx + fw / 2, 274, fx + fw / 2, 290)
s.arrow(sx + 60, 370, fx, 200, elbow=490, label=None)
s.text(492, 300, "B_nav (values)", size=9)
s.text(fx + fw / 2, 372, "chunked over 4096 queries; O(N x K) instead of O(N^2)", size=8.5, anchor="middle")
s.text(fx + fw / 2, 386, "N = 60,000 BEV cells", size=8.5, anchor="middle")
s.box(fx, 400, fw, 56, "Route reconstruction head",
      ["Conv1x1 256->64, GN, SiLU, dw 3x3, SiLU, Conv1x1 64->2", "bilinear to 450 x 300; 0.017 M"], style="aux")
s.arrow(rx + rw, 338, fx, 428, elbow=505, color="#B8860B")
s.legend(20, 462, [("input", "input"), ("frozen", "frozen"), ("trainable", "trainable"), ("aux", "auxiliary")])
s.save(OUT + "fig06_navigation_fusion.svg")

# ======================= Fig 7: GRU planner =======================
s = SVG(760, 330, font_size=10.5)
s.text(14, 22, "Deterministic GRU control planner (0.835 M params) and unicycle rollout", size=11, weight="bold")
s.box(20, 40, 200, 70, "Context c_0 (256-d)", ["P_ego(h_ego): Linear 256 -> 256", "+ P_vis(h_vis): Linear 896 -> 256",
      "h_vis = 0 in evaluated checkpoints"], style="input")
s.box(20, 126, 200, 56, "Values V = value_proj(B_fused)", ["Linear 256 -> 256 per cell", "256 x 300 x 200"], style="trainable")
s.box(20, 198, 200, 54, "Learned ego query e (256)", ["nn.Embedding(1, 256)"], style="trainable")
# unrolled cells
cx0 = 260
for i, lab in enumerate(["t = 1", "t = 2", "...", "t = 64"]):
    cx = cx0 + i * 122
    if lab == "...":
        s.text(cx + 50, 150, "...", size=16, anchor="middle")
        continue
    s.box(cx, 60, 108, 150, f"step {lab}",
          ["q = detach(h) + e", "ref = sigmoid(W_r q)", "off = W_o q (16 x 2)", "loc = ref + 0.1 off",
           "a = sum_k w_k V[loc_k]", "a = W_out a", "h = GRU(a, h)", "u = W_u h (2)"], style="trainable", line_size=8.5)
s.arrow(220, 75, cx0, 75, label="h_0 = c_0", label_dy=-4)
s.arrow(cx0 + 108, 140, cx0 + 122, 140, label="h", label_dy=-4)
s.arrow(cx0 + 122 * 3 - 14, 140, cx0 + 122 * 3, 140, label="h", label_dy=-4)
s.arrow(220, 154, 240, 154); s.text(232, 168, "V shared by all steps (bilinear grid_sample, 16 points, zero padding)", size=8.5)
s.arrow(220, 225, 240, 225); s.text(232, 240, "e shared by all steps", size=8.5)
s.box(260, 232, 474, 40, "u_1..u_64 = (a_t, kappa_t) @ 10 Hz -> semi-implicit unicycle rollout (dt = 0.1 s, float32)",
      ["v_t = max(v_{t-1} + a_t dt, 0);  theta_t = sum v_i kappa_i dt;  x_t = sum v_i cos(theta_i) dt;  y_t = sum v_i sin(theta_i) dt"],
      style="output", line_size=8.8, mono_lines=("v_t = max(v_{t-1} + a_t dt, 0);  theta_t = sum v_i kappa_i dt;  x_t = sum v_i cos(theta_i) dt;  y_t = sum v_i sin(theta_i) dt",))
for i in (0, 1, 3):
    s.arrow(cx0 + i * 122 + 54, 210, cx0 + i * 122 + 54, 232)
s.text(20, 296, "The recurrent state is detached before the deformable lookup, so BEV-lookup gradients do not flow through the recurrence;",
       size=9)
s.text(20, 310, "the same 16-point lookup is applied at every step. Sampling locations are clamped to the BEV extent.", size=9)
s.save(OUT + "fig07_gru_planner.svg")

# ======================= Fig 8: lineage + evaluation protocol =======================
s = SVG(780, 470, font_size=10.5)
s.text(14, 22, "Checkpoint lineage (top) and the two evaluation protocols (bottom)", size=11, weight="bold")
s.box(14, 40, 176, 76, "Official BEVFormer V2 R50 T8",
      ["nuScenes-trained (epoch 24)", "BEV queries/pos. embeddings resized", "to 300 x 200; detector heads dropped"], style="frozen")
s.box(214, 40, 176, 76, "nuPlan trajectory stage",
      ["8 GPU workers, seed 149, LR 1e-4", "camera BEV frozen; traj w=1, route w=1", "BEV seg w=0; 1,024 internal val samples"],
      style="trainable")
s.box(414, 40, 156, 76, "nuPlan Epoch 4 / Epoch 5",
      ["ed00e072471a / ca8b43d7a777", "Epoch 5 = fine-tuning parent", "(frozen Epoch-5 parent profile)"], style="plain")
s.box(594, 40, 172, 76, "KITScenes fine-tuning",
      ["8 workers, seed 149, LR 3e-5", "38,847 samples / 364 scenes", "frozen val 3,820 / 40 scenes; BF16"], style="trainable")
s.arrow(190, 78, 214, 78); s.arrow(390, 78, 414, 78); s.arrow(570, 78, 594, 78)
s.box(594, 132, 172, 52, "KITScenes Epoch 5 / Epoch 7",
      ["120a21639d97 / a1e6b1621018", "Ep. 5 marked best retained checkpoint"], style="plain")
s.arrow(680, 116, 680, 132)
s.text(214, 130, "Optional BEV-only Stage A (unfrozen camera BEV, BEV seg. loss) exists in the code path;", size=8.5, color="#555")
s.text(214, 142, "its use as parent of the reported nuPlan checkpoints is not recorded in the available metadata.", size=8.5, color="#555")
s.text(14, 130, "all four checkpoints:", size=8.5, color="#555"); s.text(14, 142, "eval_gate_pass = false", size=8.5, color="#555")

# bottom: protocols
y0 = 206
s.text(14, y0 - 6, "Protocol A: checkpoint-internal validation (model executed)", size=10.5, weight="bold")
s.box(14, y0, 236, 96, "Frozen internal validation splits",
      ["nuPlan: 1,024 samples", "KITScenes: 3,820 samples / 40 scenes", "Camera + Map + Route (+ counterfactuals)",
       "BF16 autocast"], style="input")
s.box(270, y0, 236, 96, "Metrics @ 6.4 s (64 steps)",
      ["ADE/FDE 6.4 s; route reconstruction IoU", "route corridor compliance/success",
       "route progress proxy; drivable compliance", "comfort (nuPlan thresholds)"], style="aux")
s.arrow(250, y0 + 48, 270, y0 + 48)
s.text(14, y0 + 112, "Not comparable with Protocol B: different samples, horizon, and metric definitions.", size=8.5, color="#B8443F")

y1 = 340
s.text(14, y1 - 6, "Protocol B: external deterministic overlay replay (model NOT re-executed)", size=10.5, weight="bold")
s.box(14, y1, 236, 96, "KITScenes Val v3.5 (primary)",
      ["117 scenes, 11,035 samples (strict identity)", "inputs: Camera + HD Map + Route",
       "Dashboard shows 140 shards / 13,525 samples"], style="input")
s.box(270, y1, 236, 96, "KITScenes Test v1.0 (camera-only track)",
      ["206 scenes, 23,690 samples", "Map and Route unavailable by construction",
       "different scene population from Val"], style="input")
s.box(526, y1, 240, 96, "Replay of published control overlays",
      ["64 x (a, kappa) + v0 per sample -> rollout", "ADE/FDE @ 1, 2, 3, 5 s; |lat|, |lon| error",
       "50 valid steps (78.125 %); 0 nonfinite", "no counterfactuals / gradients / route logits"], style="aux")
s.arrow(250, y1 + 48, 270, y1 + 48, color="#8A8A8A", marker="arrow-gray", dash="3,3")
s.arrow(506, y1 + 48, 526, y1 + 48)
s.text(260, y1 + 112, "Val and Test are different scene populations: Test minus Val is not a Map/Route effect.", size=8.5, color="#B8443F")
s.legend(526, y0 + 130, [("frozen", "pretrained init"), ("trainable", "training stage"), ("input", "data"), ("aux", "metrics")])
s.save(OUT + "fig08_lineage_protocol.svg")

# ======================= Fig 14: route provenance =======================
s = SVG(760, 250, font_size=10.5)
s.text(14, 22, "How the KITScenes HD Map and Route rasters are produced (scene-level, a-posteriori route)", size=11, weight="bold")
s.box(14, 40, 170, 90, "Lanelet2 map (map.osm)",
      ["lanelet polygons -> drivable / intersection", "left/right bounds -> lane boundary", "centerlines -> centerline + direction",
       "crosswalk subtype; stop lines; traffic lights"], style="input", line_size=8.6)
s.box(204, 40, 170, 90, "Recorded ego trace (whole scene)",
      ["10 Hz poses (x, y, yaw)", "used ONLY for route matching", "no future ego points are rasterised"], style="input", line_size=8.6)
s.box(394, 40, 170, 90, "Lanelet2 trace matcher",
      ["candidate radius 8 m, dist. sigma 2 m", "heading sigma 20 deg; transition costs", "same 0 / following 0.25 / adjacent 1 /",
       "disconnected 25; quality thresholds"], style="plain", line_size=8.6)
s.box(584, 40, 162, 90, "Route = driven lane sequence",
      ["lane-level corridor (3.5 m wide)", "destination = last scene pose (r = 2 m)", "confidence & quality metadata",
       "route_valid may be false"], style="output", line_size=8.6)
s.arrow(184, 85, 204, 85); s.arrow(374, 85, 394, 85); s.arrow(564, 85, 584, 85)
s.box(14, 156, 350, 70, "Native C++ rasteriser (450 x 300 @ 0.4 m, ego frame)",
      ["rendered at 500 ms anchor poses (latest non-future pose),", "then SE(2)-warped to the exact sample pose",
       "14 map channels in [0,1] + 2 binary route channels"], style="plain", line_size=8.6)
s.box(394, 156, 352, 70, "Implication for interpretation",
      ["the route is the lane sequence the human actually drove;", "it is an oracle-quality navigation intent, not a live planner output;",
       "route reconstruction IoU measures preservation of this input"], style="note", line_size=8.6)
s.arrow(99, 130, 99, 156); s.arrow(665, 130, 665, 156)
s.save(OUT + "fig14_route_provenance.svg")

# ======================= Fig 15: matched ablation design =======================
s = SVG(760, 300, font_size=10.5)
s.text(14, 22, "Required (not yet run) same-scene matched ablation for a causal Map/Route claim", size=11, weight="bold")
s.box(14, 44, 220, 70, "Fixed inputs for all conditions",
      ["same 11,035 KITScenes Val samples", "same frozen checkpoint (e.g. KITScenes Ep. 5)",
       "identical precomputed camera BEV B_img"], style="input")
for i, (lab, lines, st) in enumerate([
    ("A: Camera only", ["map_valid = 0, route_valid = 0", "B_nav = E_map(0) + 0"], "plain"),
    ("B: Camera + HD Map", ["map_valid = 1, route_valid = 0", "(enable_route_conditioning off)"], "plain"),
    ("C: Camera + Map + Route", ["map_valid = 1, route_valid = 1", "= production contract"], "plain")]):
    x = 262 + i * 166
    s.box(x, 44, 150, 70, lab, lines, style=st, line_size=8.8)
    s.arrow(234, 79, x, 79) if i == 0 else None
s.arrow(412, 79, 428, 79); s.arrow(578, 79, 594, 79)
s.box(14, 136, 730, 60, "Report per condition (paired, scene-level bootstrap CIs, distribution of per-scene deltas)",
      ["ADE/FDE @ 1, 2, 3, 5 s; mean |lat| / |lon| error; route corridor compliance & success; route progress proxy;",
       "drivable compliance & success; comfort rate and each violation; failure rate & nonfinite predictions; latency/memory"],
      style="aux", line_size=8.8)
s.box(14, 214, 730, 62, "Controls and stratifications",
      ["shuffled-route control (another scene's route); destination-marker removal; route-corridor removal; map-only corruption/dropout;",
       "off-route / ambiguous-intersection subsets; valid-map vs invalid-map stratification"], style="note", line_size=8.8)
s.text(14, 292, "No outcome of this experiment is reported in this paper.", size=9.5, color="#B8443F", weight="bold")
s.save(OUT + "fig15_matched_ablation.svg")
print("schematics written")
