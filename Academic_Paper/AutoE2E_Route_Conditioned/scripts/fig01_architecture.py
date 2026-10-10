# ruff: noqa
"""Figure 1: AutoE2E evaluated architecture (bevformer_v2_t8_split_navigation_v5)."""
import sys
sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
from svgkit import SVG, MONO

W, H = 1100, 724
s = SVG(W, H, font_size=11)

C0, W0 = 14, 192      # inputs
C1, W1 = 240, 150     # backbone
C2, W2 = 424, 196     # encoder / map enc
C3, W3 = 654, 176     # temporal fusion / route gate
C4, W4 = 882, 204     # fusion / planner

s.text(W - 12, 20, "79.91 M parameters in total; 1.45 M trainable (1.8%) in the trajectory stages",
       size=10.5, anchor="end", weight="bold")

# ============ Top band: camera path ============
s.text(C0 + W0 / 2, 22, "Camera inputs", size=12, weight="bold", anchor="middle")
s.box(C0, 32, W0, 72, "Camera history (7 frames)",
      ["6 views x 512x512 each", "t-3.5 s ... t-0.5 s, 0.5 s interval",
       "512 px only; detached (no grad)"], style="input")
s.box(C0, 116, W0, 50, "Current cameras (t = 0)",
      ["6 views x 512x512", "long-range front + 5 surround"], style="input")
s.box(C0, 178, W0, 50, "Native front camera (t = 0)",
      ["1 view x 1024x1024", "additive high-res branch"], style="input")

s.box(C1, 32, W1, 196, "ResNet-50 + FPN",
      ["shared for all frames/views", "BEVFormer V2 input profile",
       "BatchNorm frozen", "", "FPN: last 3 stages ->", "4 levels x 256 ch", "",
       "23.51 M + 3.28 M params"], style="frozen", badge="frozen")

s.box(C2, 32, W2, 118, "BEVFormer V2 T1 encoder x6",
      ["BEV queries 300x200x256 (15.36 M)", "+ row/col positional embeddings",
       "layer: deformable self-attn (4 pts)", "+ spatial cross-attn (8 heads x",
       "4 levels x 8 pts) + FFN 512", "view fusion 20.70 M (layers 4.94 M)"],
      style="frozen", badge="frozen")
s.box(C2, 172, W2, 56, "Front residual branch (t = 0 only)",
      ["copy of layer-6 cross-attn (0.33 M, frozen)",
       "content-only delta x tanh(gate); gate 256-d"],
      style="trainable", badge="gate train")

s.box(C3, 32, W3, 118, "T8 temporal fusion",
      ["in: 7 detached history BEVs + current", "concat 8 BEVs (t-7 ... t0) = 2048 ch",
       "3 ResNet BasicBlocks @ 512 ch", "Linear 512 -> 256 + LayerNorm; 30.81 M", "",
       "B_img: 256 x 300 x 200 (0.6 m)"], style="frozen", badge="frozen",
      mono_lines=("B_img: 256 x 300 x 200 (0.6 m)",))
s.box(C3, 236, W3, 38, "BEV segmentation head (dormant)",
      ["8 classes from B_img; loss weight 0.0"], style="dormant")

s.box(C4, 32, W4, 118, "Deformable navigation fusion",
      ["query: each B_img cell; values: B_nav",
       "K = 8 sample points, 8 heads, 256 ch",
       "pre-LN, FFN 256 -> 512 -> 256",
       "out-proj + FFN zero-initialised", "0.350 M params", "",
       "B_fused = B_img + F(B_img, B_nav)"], style="trainable", badge="train",
      mono_lines=("B_fused = B_img + F(B_img, B_nav)",))

# arrows: cameras -> backbone
s.arrow(C0 + W0, 68, C1, 68)
s.arrow(C0 + W0, 141, C1, 141)
s.arrow(C0 + W0, 203, C1, 203)
# backbone -> encoder and front branch
s.arrow(C1 + W1, 90, C2, 90, label="8 x FPN pyramids", label_dy=-4)
s.arrow(C1 + W1, 200, C2, 200, label="front FPN", label_dy=-4)
# encoder -> temporal fusion: history BEVs
s.arrow(C2 + W2, 60, C3, 60)
# encoder current BEV -> front branch
s.arrow(C2 + 36, 150, C2 + 36, 172)
s.text(C2 + 41, 165, "current BEV", size=8.5)
# front branch -> temporal fusion (enter at bottom-left of T8 box)
s.arrow(C2 + W2, 200, C3 + 24, 150, elbow=C3 + 24)
s.text(C3 + 30, 172, "current BEV", size=8.5)
s.text(C3 + 30, 182, "+ gated front delta", size=8.5)
# temporal fusion -> deformable fusion
s.arrow(C3 + W3, 90, C4, 90, label="B_img (query)", label_dy=-4)
# B_img -> dormant head
s.arrow(C3 + W3 - 24, 150, C3 + W3 - 24, 236, color="#8A8A8A", marker="arrow-gray", dash="4,3")

# ============ Middle band: navigation ============
s.text(C0 + W0 / 2, 292, "Navigation inputs", size=12, weight="bold", anchor="middle")
s.box(C0, 302, W0, 74, "HD Map raster M",
      ["14 ch x 450x300 @ 0.4 m/px", "same extent as camera BEV",
       "values in [0,1]; map_valid gate"], style="input")
s.box(C0, 388, W0, 62, "Route raster R",
      ["2 ch x 450x300, binary", "corridor 3.5 m / destination r = 2 m",
       "route_valid gate"], style="input")

s.box(C2, 302, W2, 74, "Map encoder E_map",
      ["fully convolutional semantic raster enc.",
       "5x5 conv 96 -> dw-stride 2 & 4 -> 1x1 256",
       "resample to 300x200; 0.167 M"], style="trainable", badge="train")
s.box(C2, 388, W2, 62, "Route encoder E_route",
      ["lightweight conv, 96 hidden ch", "keeps thin corridor/destination",
       "0.078 M"], style="trainable", badge="train")
s.box(C3, 388, W3, 62, "Route gate",
      ["sigmoid(g), g in R^256, zero-init", "route_contribution = gate * E_route(R)"],
      style="trainable", badge="train")
sx, sy = C3 + W3 / 2, 339
s.add(f'<circle cx="{sx}" cy="{sy}" r="12" fill="white" stroke="#333" stroke-width="1.3"/>')
s.text(sx, sy + 4.5, "+", size=15, anchor="middle", weight="bold")
s.text(sx, sy - 20, "B_nav = E_map(M) + gate * E_route(R)", size=9.5, anchor="middle",
       weight="bold", family=MONO)

s.arrow(C0 + W0, 339, C2, 339)
s.arrow(C0 + W0, 419, C2, 419)
s.arrow(C2 + W2, 339, sx - 12, 339)
s.arrow(C2 + W2, 419, C3, 419)
s.arrow(sx, 388, sx, sy + 12)
# B_nav -> fusion (values): right then up into fusion bottom-left lane
bn_x = C4 + 44
s.arrow(sx + 12, 339, bn_x, 150, elbow=bn_x)
s.text(bn_x + 6, 250, "B_nav (values)", size=9.5)

# ============ Right column: planner ============
s.box(C4, 302, W4, 128, "GRU control planner",
      ["c_0 = P_ego(h_ego) + P_vis(h_vis), 256-d",
       "for t = 1..64:",
       "  q_t = detach(h_{t-1}) + e_query",
       "  a_t = deformable lookup (16 pts) on B_fused",
       "  h_t = GRU(a_t, h_{t-1});  u_t = W_u h_t",
       "u_t = (acceleration, curvature); 0.835 M"],
      style="trainable", badge="train")
bf_x = C4 + W4 - 44
s.arrow(bf_x, 150, bf_x, 302, label=None)
s.text(bf_x + 6, 230, "B_fused", size=9.5)
s.text(bf_x + 6, 242, "(value_proj)", size=8.5)

s.box(C4, 452, W4, 58, "Semi-implicit unicycle rollout",
      ["dt = 0.1 s, v_0 = current speed",
       "v, heading, XY integrated in float32"], style="plain")
s.arrow(C4 + W4 / 2, 430, C4 + W4 / 2, 452)
s.text(C4 + W4 / 2 + 8, 445, "u_1..u_64 (64 x 2)", size=9)
s.box(C4, 532, W4, 52, "Ego-frame trajectory",
      ["64 x 2 (XY) @ 10 Hz = 6.4 s horizon"], style="output")
s.arrow(C4 + W4 / 2, 510, C4 + W4 / 2, 532)

s.box(C4, 604, W4, 60, "Trajectory imitation loss (w = 1.0)",
      ["Smooth-L1 (beta 1 m) on rolled-out XY", "vs recorded XY, validity-masked"],
      style="loss")
s.arrow(C4 + W4 / 2, 584, C4 + W4 / 2, 604, color="#B1407A")

# ============ Bottom-left: ego / visual history ============
s.text(C0 + W0 / 2, 510, "History inputs", size=12, weight="bold", anchor="middle")
s.box(C0, 520, W0, 60, "Egomotion history h_ego",
      ["64 steps x 4 @ 10 Hz = 6.4 s", "speed/33, accel/8, yaw rate, curvature"],
      style="input")
s.box(C0, 626, W0, 48, "Visual history h_vis",
      ["896-d API input; zeros (no_memory)"], style="dormant")
s.arrow(C0 + W0, 550, C4, 400, elbow=848, color="#333")
s.text(500, 545, "h_ego -> P_ego (Linear 256 -> 256)", size=9.5, anchor="middle")
s.arrow(C0 + W0, 650, C4, 416, elbow=864, color="#8A8A8A", marker="arrow-gray", dash="4,3")
s.text(500, 645, "h_vis -> P_vis (Linear 896 -> 256); input is zero", size=9.5,
       anchor="middle", color="#777")

# ============ Aux + losses (bottom center) ============
s.box(C3, 480, W3, 60, "Route reconstruction head",
      ["input: route_contribution", "2 ch logits @ 450x300; 0.017 M"],
      style="aux", badge="train")
s.arrow(sx, 450, sx, 480, color="#B8860B", label="route_contribution", label_dy=-4,
        label_size=8.5)
s.box(C3, 560, W3, 60, "Route reconstruction loss (w = 1.0)",
      ["corridor: 0.5 BCE + 0.5 soft Dice", "destination: 0.25 x focal heatmap"],
      style="loss")
s.arrow(sx, 540, sx, 560, color="#B1407A")
s.text(C3 + W3 / 2, 690, "BEV segmentation loss weight = 0.0 in all four reported checkpoints",
       size=9.5, anchor="middle", color="#555")

# ============ Legend ============
s.legend(14, 712, [("input", "data input"), ("frozen", "frozen (official BEVFormer V2 T8 init.)"),
                   ("trainable", "trainable in trajectory stages"),
                   ("dormant", "present, disabled"), ("aux", "auxiliary head"),
                   ("output", "output"), ("loss", "loss")])

s.save("/tmp/autoe2e_paper/vivlio/figures/fig01_architecture.svg")
print("ok")
