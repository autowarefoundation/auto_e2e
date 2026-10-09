# ruff: noqa
"""Polished replacements for figures 2 and 3."""
import sys
sys.path.insert(0, "/tmp/autoe2e_paper/figgen")
from svgkit import SVG, MONO
OUT="/tmp/autoe2e_paper/vivlio/figures/"

# ---------- Temporal contract ----------
W,H=900,360
s=SVG(W,H,font_size=10.5)
s.text(18,22,"Temporal contract: camera history, egomotion, control output, and benchmark targets",size=11,weight="bold")
x0,x1=190,870
def tx(t): return x0+(t+6.4)/12.8*(x1-x0)
# now line
s.line(tx(0),35,tx(0),312,color="#d62728",dash="4,3",width=1.2)
s.text(tx(0)+6,48,"now (t = 0)",size=9,color="#d62728")
# Camera row
s.text(18,82,"Camera (T8)",size=10.5,weight="bold")
for t in [-3.5,-3,-2.5,-2,-1.5,-1,-.5,0]:
    cur=t==0; x=tx(t)
    s.add(f'<rect x="{x-7}" y="58" width="14" height="32" rx="2" fill="{"#FBE5E3" if cur else "#E3EEF9"}" stroke="{"#d62728" if cur else "#3F6FAE"}" stroke-width="1.2"/>')
    s.text(x,103,"0" if cur else f"{t:.1f}",size=7.5,anchor="middle",color="#d62728" if cur else "#3F6FAE")
s.text(x0,50,"7 history frames x 6 views @ 512 px; detached",size=8.6,color="#3F6FAE")
s.text(tx(0)+18,72,"6 views @ 512 px",size=8.6,color="#d62728")
s.text(tx(0)+18,84,"+ front @ 1024 px (additive)",size=8.6,color="#d62728")
s.text(tx(0)+18,96,"3.5 s visual span at 2 Hz",size=8.6,color="#333")
# Egomotion row
s.text(18,153,"Egomotion",size=10.5,weight="bold")
s.add(f'<rect x="{tx(-6.4)}" y="126" width="{tx(-4)-tx(-6.4)}" height="34" rx="4" fill="#F5F5F5" stroke="#999"/>')
s.text((tx(-6.4)+tx(-4))/2,147,"zero pad in benchmark",size=8.4,anchor="middle",color="#666")
s.add(f'<rect x="{tx(-4)}" y="126" width="{tx(0)-tx(-4)}" height="34" rx="4" fill="#E3EEF9" stroke="#3F6FAE"/>')
s.text((tx(-4)+tx(0))/2,147,"40 real steps (4.0 s) in benchmark",size=8.6,anchor="middle",weight="bold")
s.text(x0,119,"64 x [speed, acceleration, yaw rate, curvature] @ 10 Hz; training uses all 6.4 s",size=8.5)
# Planner output row
s.text(18,219,"Planner output",size=10.5,weight="bold")
s.add(f'<rect x="{tx(0)}" y="191" width="{tx(6.4)-tx(0)}" height="34" rx="4" fill="#E2F3E4" stroke="#2E8B57"/>')
s.text((tx(0)+tx(6.4))/2,212,"64 x (acceleration, curvature) @ 10 Hz -> rollout -> 64 XY points",size=8.6,anchor="middle",weight="bold")
# Target row
s.text(18,276,"Targets",size=10.5,weight="bold")
s.add(f'<rect x="{tx(0)}" y="248" width="{tx(5)-tx(0)}" height="34" rx="4" fill="#FCF4DB" stroke="#B8860B"/>')
s.text((tx(0)+tx(5))/2,269,"50 valid steps: ADE/FDE @ 1, 2, 3, 5 s; |lateral| / |longitudinal|",size=8.4,anchor="middle",weight="bold")
s.add(f'<rect x="{tx(5)}" y="248" width="{tx(6.4)-tx(5)}" height="34" rx="4" fill="#F5F5F5" stroke="#999"/>')
s.text((tx(5)+tx(6.4))/2,269,"no target",size=8.5,anchor="middle",color="#666")
s.text(tx(5)-4,241,"valid coverage 50/64 = 78.125%",size=8.5,anchor="end",color="#B8860B")
# Axis
ay=312;s.line(x0,ay,x1,ay,color="#333",width=1.1)
for t in range(-6,7):
    s.line(tx(t),ay-4,tx(t),ay+4,color="#333")
    s.text(tx(t),ay+17,"0" if t==0 else f"{t:+d}",size=8.5,anchor="middle")
s.text((x0+x1)/2,ay+36,"time relative to current frame [s]  (10 Hz base rate = 0.1 s per step)",size=9,anchor="middle")
s.text(W-12,22,"benchmark: 40 history / 50 future steps; training: 64 / 64",size=9.5,anchor="end",color="#555")
s.save(OUT+"fig02_temporal_contract.svg")

# ---------- BEVFormer encoder layer and front residual ----------
W,H=900,510
s=SVG(W,H,font_size=10.5)
s.text(18,22,"BEVFormer V2 encoder layer (x6, frozen) and the current-frame high-resolution front residual",size=11,weight="bold")
# left chain
x,w=30,330
s.box(x,42,w,48,"BEV queries Q  (60,000 x 256)",["300 x 200 learned embeddings + row/column position (128+128)"],style="frozen")
s.box(x,112,w,82,"Temporal self-attention (T1 form)",["deformable; queue = {Q, Q} (current BEV duplicated)","8 heads x 4 points per queue; offsets from [Q, Q+pos]","value_proj / output_proj 256 x 256; 0.23 M"],style="frozen")
s.box(x,216,w,112,"Multi-scale spatial cross-attention",["4 pillar heights z in [-5, 3] m -> calibrated pinhole/fisheye projection","8 heads x 4 FPN levels x 8 points per visible view","visible-view average + level/camera embeddings; 0.33 M"],style="frozen")
s.box(x,350,w,58,"FFN 256 -> 512 -> 256",["ReLU, dropout 0.1; 0.26 M; three post-norm LayerNorms"],style="frozen")
s.box(x,430,w,44,"Layer-6 output B_enc (current BEV)",["also passed to T8 temporal fusion"],style="plain")
for a,b in [(90,112),(194,216),(328,350),(408,430)]: s.arrow(x+w/2,a,x+w/2,b)
for yy in [155,270,378]: s.text(x+w+10,yy,"LN",size=8.5,color="#555")
# FPN input
s.box(390,112,190,82,"Camera FPN features",["6 views x 4 levels x 256 ch","512 px base tiles; history pyramids cached"],style="input")
s.arrow(485,194,360,260,elbow=485,label="values",label_dy=-4)
# right front branch
rx,rw=610,260
s.box(rx,42,rw,62,"Native front tile 1024 x 1024 (t = 0)",["shared ResNet-50 + FPN; 4 levels, one view"],style="input")
s.box(rx,126,rw,112,"Front cross-attention (frozen copy of layer 6)",["reference points from the calibrated front-camera operator","content-only delta: no level/camera embedding affine terms","delta = LN6(Q + attn) - LN6(Q); masked to observed cells"],style="frozen")
s.box(rx,260,rw,62,"Gate tanh(g), g in R^256, zero-initialised",["only trainable camera-side parameter: 256 values"],style="trainable",badge="gate train")
s.box(rx,344,rw,64,"B_0 = B_enc + tanh(g) * delta",["at initialisation: B_0 = B_enc exactly (no-op)"],style="plain",mono_lines=("B_0 = B_enc + tanh(g) * delta",))
s.box(rx,430,rw,44,"T8 fusion input: B_0 + 7 detached history BEVs",[],style="frozen")
for a,b in [(104,126),(238,260),(322,344),(408,430)]: s.arrow(rx+rw/2,a,rx+rw/2,b)
# current BEV feeds front branch query / residual path
s.arrow(360,452,610,376,label="B_enc",label_dy=-4)
# visual styling legend
s.legend(30,500,[("input","input"),("frozen","frozen"),("trainable","trainable")])
s.text(570,500,"The 1024-pixel branch is used only for t=0; every historical front image remains 512 x 512.",size=8.5,anchor="middle",color="#555")
s.save(OUT+"fig03_encoder_layer.svg")
print("fixed figures written")
