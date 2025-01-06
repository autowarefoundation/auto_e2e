"""Architecture-diagram renderer (arch mode) — dependency-free, left-to-right.

Takes an arch_v1 IR (schema/arch_v1.schema.json) and lays it out automatically into a
left-to-right diagram in the style of a hand-drawn paper/README architecture figure:
inputs on the LEFT, the data spine through the middle, outputs + a pink loss column on the
RIGHT, branching/merging, dashed loss/feedback edges, and an "ONLY DURING TRAINING" banner.

Pure stdlib + Python-generated inline SVG (no JS libs, no graphviz, no CDN). The output is a
self-contained HTML shell (dark theme, Save-PNG button, click-to-detail tip) defined here.

Layout = a small deterministic Sugiyama-lite pipeline:
  1. layering (x): longest-path over dataflow edges; pin inputs left, outputs right, losses
     in a dedicated far-right column; pull-right tightening; honor optional lane hints.
  2. row ordering (y): barycenter sweeps, keep lowest-crossing; honor optional row hints.
  3. coordinates: variable box heights from estimated text wrapping; per-column centering.
  4. edges: bezier forward edges with spread ports; feedback edges bow through a reserved
     top channel; loss=pink dashed, skip=thin gold, feedback=amber dashed.
  5. group banners: bounding band + pink pill over train_only members.
All ordering keys are total (tie-break on IR index) so the output is byte-stable.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .validate import _has_cycle


def _esc(s: Any) -> str:
    return html.escape(str(s)) if s is not None else ""


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# base role -> (fill, stroke) palette; the 5 arch-specific roles are overlaid in ARCH_COLORS
ROLE_COLORS = {
    "input":              ("#10233a", "#2f6fb0"),
    "output":             ("#3a1622", "#b83a3e"),
    "backbone":           ("#0e2a26", "#12a594"),
    "embedding":          ("#0e2a26", "#12a594"),
    "convolution":        ("#2a2016", "#ad7f58"),
    "self_attention":     ("#3a1718", "#e5484d"),
    "cross_attention":    ("#3a2410", "#f5a623"),
    "linear_proj":        ("#23252c", "#8b8d98"),
    "ffn_mlp_block":      ("#261a3a", "#8e4ec6"),
    "moe_block":          ("#261a3a", "#8e4ec6"),
    "normalization":      ("#0d2236", "#0091ff"),
    "activation":         ("#10241a", "#30a46c"),
    "positional_encoding": ("#2e1228", "#d6409f"),
    "recurrent":          ("#2a2410", "#c9a227"),
    "conditioning":       ("#3a2410", "#f5a623"),
    "pooler":             ("#23252c", "#8b8d98"),
    "fusion":             ("#2a2410", "#ffb224"),
    "head":               ("#3a1622", "#b83a3e"),
    "merge_add":          ("#23252c", "#c8cad0"),
    "buffer":             ("#1c2128", "#8896a6"),
    "other":              ("#1c2128", "#6e7681"),
}

# self-contained HTML shell (dark theme, Save-PNG button, click-to-detail tip). The inline
# SVG keeps id="flow" so this template's JS finds it. No JS libraries, no CDN.
_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root{color-scheme:dark}
  html,body{margin:0;background:#0b0d12;color:#e8eaed;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #hdr{padding:14px 18px 6px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
  #hdr h1{margin:0;font-size:16px}
  #hdr .sub{font-size:12px;color:#9aa0ac;margin-top:3px}
  #btnpng{background:#1b1f2a;color:#c2c6cc;border:1px solid #2a3142;border-radius:6px;
    padding:6px 12px;font-size:12px;cursor:pointer;white-space:nowrap}
  #btnpng:hover{background:#232838}
  #wrap{padding:0 18px 40px;overflow:auto}
  #flow{max-width:100%;height:auto}
  .stage{cursor:pointer}
  .stage:hover rect{filter:brightness(1.18)}
  #legend{padding:6px 18px 16px;font-size:11px;color:#9aa0ac;display:flex;flex-wrap:wrap;gap:14px}
  #legend .row{display:flex;align-items:center;gap:5px}
  #legend .sw{width:11px;height:11px;border-radius:3px}
  #tip{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);
    max-width:680px;background:rgba(17,20,28,.97);border:1px solid #2a3142;border-radius:8px;
    padding:10px 14px;font-size:12.5px;line-height:1.5;display:none;box-shadow:0 6px 24px rgba(0,0,0,.5)}
  #tip b{color:#fff}
</style></head>
<body>
<div id="hdr"><div><h1>__TITLE__</h1><div class="sub">__SUB__</div></div>
  <button id="btnpng" title="Save this diagram as a PNG image">&#11015; Save PNG</button></div>
<div id="legend">__LEGEND__</div>
<div id="wrap">__SVG__</div>
<div id="tip"></div>
<script>
  var tip=document.getElementById('tip');
  document.querySelectorAll('.stage').forEach(function(g){
    g.addEventListener('click',function(){
      var d=g.getAttribute('data-detail');
      var t=g.querySelector('text');
      if(!d){tip.style.display='none';return;}
      tip.innerHTML='<b>'+(t?t.textContent:'')+'</b><br>'+d;
      tip.style.display='block';
    });
  });
  document.addEventListener('keydown',function(e){if(e.key==='Escape')tip.style.display='none';});

  // Save-PNG: rasterize the inline SVG to a 2x canvas and download.
  document.getElementById('btnpng').addEventListener('click',function(){
    var svg=document.getElementById('flow');
    var vb=svg.viewBox.baseVal, W=vb&&vb.width?vb.width:svg.clientWidth, H=vb&&vb.height?vb.height:svg.clientHeight;
    var scale=2;
    var data=new XMLSerializer().serializeToString(svg);
    if(data.indexOf('xmlns=')===-1) data=data.replace('<svg','<svg xmlns="http://www.w3.org/2000/svg"');
    var blob=new Blob([data],{type:'image/svg+xml;charset=utf-8'});
    var url=URL.createObjectURL(blob);
    var img=new Image();
    img.onload=function(){
      var c=document.createElement('canvas'); c.width=W*scale; c.height=H*scale;
      var ctx=c.getContext('2d'); ctx.fillStyle='#0b0d12'; ctx.fillRect(0,0,c.width,c.height);
      ctx.scale(scale,scale); ctx.drawImage(img,0,0);
      URL.revokeObjectURL(url);
      c.toBlob(function(b){
        var a=document.createElement('a');
        a.download=(document.title.replace(/[^a-z0-9]+/gi,'_'))+'.png';
        a.href=URL.createObjectURL(b); a.click();
      });
    };
    img.src=url;
  });
</script>
</body></html>
"""

# ---- palette: 5 new figure roles overlaid on the shared flow palette ----
ARCH_COLORS = {
    "loss":            ("#3a1326", "#e06c9a"),  # pink — Trajectory/Feature Loss
    "policy":          ("#261a3a", "#8e4ec6"),  # purple — Driving Policy (fan-out hub)
    "prediction":      ("#10241a", "#30a46c"),  # green — predicted future features/waypoints
    "future_state":    ("#161d3a", "#5b7fff"),  # indigo — train-only Future Visual State
    "learning_method": ("#23252c", "#8b8d98"),  # slate — Imitation / Reinforcement Learning
}

EDGE_STYLE = {
    "dataflow": {"color": "#5b6472", "dash": None,   "width": 2.0, "marker": "ar-data"},
    "loss":     {"color": "#e06c9a", "dash": "6 4",  "width": 1.8, "marker": "ar-loss"},
    "feedback": {"color": "#c9a227", "dash": "5 4",  "width": 1.6, "marker": "ar-feedback"},
    "skip":     {"color": "#ffe08a", "dash": "4 4",  "width": 1.5, "marker": "ar-skip"},
}

# ---- geometry constants (dark theme, tuned for paper-figure readability) ----
BOX_W = 212
COL_GAP = 104
ROW_GAP = 26
MIN_H = 56
PAD_L = 36
PAD_R = 40
PAD_BOT = 52
LINE_H = 15
PX = 14            # inner horizontal padding
TITLE_H = 20
SHAPE_H = 15
FEEDBACK_CH = 26   # height of one feedback lane in the top channel
BANNER_H = 22


def _node_color(role: str) -> tuple[str, str]:
    return ARCH_COLORS.get(role) or ROLE_COLORS.get(role) or ROLE_COLORS["other"]


def _fmt_shape_arch(shape) -> str:
    if shape is None or shape == "":
        return ""
    if isinstance(shape, list):
        return "(" + ",".join(str(x) for x in shape) + ")"
    s = str(shape).strip()
    if s and not s.startswith("("):
        s = "(" + s + ")" if ("," in s or s[0].isdigit()) else s
    return s


# ---- Python text-width estimation (no browser measureText) ----
_NARROW = set("iljftI.,:;|!'\" ")
_WIDE = set("mwMW@")


def est_text_width(s: str, px: float, bold: bool = False, mono: bool = False) -> float:
    if mono:
        return len(s) * px * 0.60
    total = 0.0
    for ch in s:
        if ch in _NARROW:
            total += 0.30
        elif ch in _WIDE:
            total += 0.92
        elif ch.isdigit() or ch.islower():
            total += 0.52
        elif ch.isupper():
            total += 0.66
        else:
            total += 0.55
    total *= px
    return total * 1.06 if bold else total
