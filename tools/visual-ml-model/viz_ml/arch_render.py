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


def _wrap(text: str, max_w: float, px: float, max_lines: int = 2) -> tuple[list[str], bool]:
    """Greedy word-wrap to <=max_lines. Returns (lines, truncated)."""
    if not text:
        return [], False
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if est_text_width(trial, px) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    truncated = False
    if len(lines) >= max_lines:
        # was there leftover content?
        joined = " ".join(lines)
        if joined.rstrip() != text.rstrip():
            truncated = True
            last = lines[-1]
            while last and est_text_width(last + "…", px) > max_w:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"
    return lines, truncated


def _box_height(node: dict) -> tuple[int, list[str], bool]:
    desc = node.get("desc", "") or ""
    lines, trunc = _wrap(desc, BOX_W - 2 * PX, 11.5, max_lines=2)
    has_shape = bool(_fmt_shape_arch(node.get("shape")))
    h = 10 + TITLE_H + len(lines) * LINE_H + (SHAPE_H if has_shape else 0) + 12
    return max(MIN_H, h), lines, trunc


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def _layout(arch: dict[str, Any]) -> dict[str, Any]:
    nodes = arch.get("nodes", [])
    edges = arch.get("edges", [])
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    by_id = {n["id"]: n for n in nodes}
    warnings: list[str] = []

    # --- Stage 1: layering over dataflow edges ---
    df = [(e["from"], e["to"]) for e in edges
          if e.get("kind") == "dataflow" and e.get("from") in by_id and e.get("to") in by_id]
    succ: dict[str, list[str]] = {nid: [] for nid in by_id}
    pred: dict[str, list[str]] = {nid: [] for nid in by_id}
    # drop a back edge if dataflow has a cycle (keep layering finite)
    adj = {nid: [] for nid in by_id}
    for a, b in df:
        adj[a].append(b)
    if _has_cycle(adj):
        warnings.append("dataflow had a cycle; some edges dropped from layering")
        # simple removal: rebuild keeping edges that don't create a cycle
        kept = []
        adj2 = {nid: [] for nid in by_id}
        for a, b in df:
            adj2[a].append(b)
            if _has_cycle(adj2):
                adj2[a].pop()
            else:
                kept.append((a, b))
        df = kept
    for a, b in df:
        succ[a].append(b)
        pred[b].append(a)

    # longest-path via Kahn topological order
    indeg = {nid: 0 for nid in by_id}
    for a, b in df:
        indeg[b] += 1
    from collections import deque
    q = deque(sorted([n for n in by_id if indeg[n] == 0], key=lambda n: idx[n]))
    topo: list[str] = []
    indeg_w = dict(indeg)
    while q:
        u = q.popleft()
        topo.append(u)
        for v in sorted(succ[u], key=lambda n: idx[n]):
            indeg_w[v] -= 1
            if indeg_w[v] == 0:
                q.append(v)
    col = {nid: 0 for nid in by_id}
    for u in topo:
        for v in succ[u]:
            col[v] = max(col[v], col[u] + 1)

    roles = {nid: by_id[nid].get("role") for nid in by_id}
    for nid in by_id:
        if roles[nid] == "input":
            col[nid] = 0

    # Forward column propagation over ALL edge kinds (skip/feedback/loss too, treated as
    # left->right for placement only — they are still excluded from the DAG layering above).
    # This pushes a node to the right of ALL its sources regardless of edge kind, so an aux
    # head fed by a `skip`, a predicted-features node fed by `dataflow` from a late branch,
    # and learning/true-feature nodes feeding a `loss` all settle into sensible columns
    # instead of being stranded in column 0. Iterate to a fixed point (acyclic over the
    # placement relation because losses/outputs are sinks).
    place_pred: dict[str, list[str]] = {nid: [] for nid in by_id}
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in by_id and b in by_id and a != b:
            place_pred[b].append(a)
    for _ in range(len(by_id) + 2):
        changed = False
        for nid in by_id:
            if roles[nid] == "input" or not place_pred[nid]:
                continue
            want = max(col[s] for s in place_pred[nid]) + 1
            if want > col[nid]:
                col[nid] = want
                changed = True
        if not changed:
            break

    # pin outputs to the last non-loss column, losses to a dedicated rightmost column
    has_loss = any(roles[n] == "loss" for n in by_id)
    nonloss_max = max([col[n] for n in by_id if roles[n] != "loss"], default=0)
    for nid in by_id:
        if roles[nid] == "output":
            col[nid] = max(col[nid], nonloss_max)
    if has_loss:
        loss_col = max(col.values()) + 1
        for nid in by_id:
            if roles[nid] == "loss":
                col[nid] = loss_col

    # pull-right tightening: non-pinned nodes with successors hug them (shortens long
    # left-anchored edges, e.g. a side input stranded far left).
    pinned = {nid for nid in by_id if roles[nid] in ("input", "output", "loss")}
    place_succ: dict[str, list[str]] = {nid: [] for nid in by_id}
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in by_id and b in by_id and a != b and col[b] > col[a]:
            place_succ[a].append(b)
    for u in reversed(topo):
        if u in pinned or not place_succ.get(u):
            continue
        col[u] = max(col[u], 0)  # never move left here; keep longest-path placement
        nxt = min(col[s] for s in place_succ[u])
        if nxt - 1 > col[u]:
            col[u] = nxt - 1

    # lane hints win
    if any(by_id[n].get("lane") is not None for n in by_id):
        for nid in by_id:
            lane = by_id[nid].get("lane")
            if lane is not None:
                col[nid] = lane

    ncols = (max(col.values()) + 1) if col else 1

    # --- Stage 2: row ordering within each column (barycenter) ---
    layers: dict[int, list[str]] = {}
    for nid in sorted(by_id, key=lambda n: idx[n]):
        layers.setdefault(col[nid], []).append(nid)

    def crossings(order: dict[int, list[str]]) -> int:
        pos = {nid: p for c in order for p, nid in enumerate(order[c])}
        cnt = 0
        for a, b in df:
            ca, cb = col[a], col[b]
            if cb != ca + 1:
                continue
        # count pairwise inversions between adjacent columns
        for c in range(ncols - 1):
            es = [(pos.get(a, 0), pos.get(b, 0)) for a, b in df if col[a] == c and col[b] == c + 1]
            for i in range(len(es)):
                for j in range(i + 1, len(es)):
                    if (es[i][0] - es[j][0]) * (es[i][1] - es[j][1]) < 0:
                        cnt += 1
        return cnt

    best = {c: list(layers.get(c, [])) for c in range(ncols)}
    best_cross = crossings(best)
    cur = {c: list(best[c]) for c in best}
    for sweep in range(4):
        down = sweep % 2 == 0
        rng = range(1, ncols) if down else range(ncols - 2, -1, -1)
        pos = {nid: p for c in cur for p, nid in enumerate(cur[c])}
        for c in rng:
            neigh = pred if down else succ
            adjc = c - 1 if down else c + 1
            apos = {nid: p for p, nid in enumerate(cur.get(adjc, []))}

            def bary(nid):
                ns = [apos[x] for x in neigh[nid] if x in apos]
                return (sum(ns) / len(ns)) if ns else pos.get(nid, 0)

            cur[c] = sorted(cur[c], key=lambda nid: (bary(nid), idx[nid]))
        cc = crossings(cur)
        if cc < best_cross:
            best_cross = cc
            best = {c: list(cur[c]) for c in cur}
    order = best

    # push train_only nodes to the BOTTOM rows of each column so the auxiliary/training
    # branch reads as a band below the main inference spine (like the reference figure).
    # Stable so the barycenter ordering is preserved within each band.
    for c in order:
        pos = {n: i for i, n in enumerate(order[c])}
        order[c] = sorted(order[c], key=lambda n: (1 if by_id[n].get("train_only") else 0, pos[n]))

    # row hints override within a column
    for c in order:
        if any(by_id[n].get("row") is not None for n in order[c]):
            order[c] = sorted(order[c], key=lambda n: (by_id[n].get("row") if by_id[n].get("row") is not None else 1e9, idx[n]))

    # --- Stage 3: coordinates ---
    heights, desclines, truncs = {}, {}, {}
    for nid in by_id:
        h, lines, tr = _box_height(by_id[nid])
        heights[nid] = h
        desclines[nid] = lines
        truncs[nid] = tr

    # count feedback lanes needed (for top channel headroom)
    fb_edges = [e for e in edges if e.get("kind") == "feedback"
                or (e.get("from") in by_id and e.get("to") in by_id and e.get("kind") == "dataflow"
                    and col[e["to"]] <= col[e["from"]])]
    n_fb = min(len(fb_edges), 4)
    pad_top = 30 + (BANNER_H + 12 if arch.get("groups") or any(by_id[n].get("train_only") for n in by_id) else 0) + n_fb * FEEDBACK_CH

    col_x = {c: PAD_L + c * (BOX_W + COL_GAP) for c in range(ncols)}
    y_top: dict[str, float] = {}
    any_train = any(by_id[n].get("train_only") for n in by_id)

    # Two-band layout: the main (inference) band on top, the train-only band below it.
    # Stack each band per column from its own cursor; the train-only band starts below the
    # tallest main-band column so the auxiliary branch reads as a clean lower region under
    # the spine (matching the reference figure), never interleaved with it.
    main_h: dict[int, float] = {}
    train_h: dict[int, float] = {}
    for c in range(ncols):
        mh = th = 0.0
        for nid in order.get(c, []):
            if by_id[nid].get("train_only"):
                th += heights[nid] + ROW_GAP
            else:
                mh += heights[nid] + ROW_GAP
        main_h[c] = max(0.0, mh - ROW_GAP)
        train_h[c] = max(0.0, th - ROW_GAP)
    main_tallest = max(main_h.values()) if main_h else 0.0
    band_gap = 40 if any_train else 0
    train_top = pad_top + main_tallest + band_gap

    for c in range(ncols):
        # main band: center each column's main stack against the tallest main column
        cursor = pad_top + (main_tallest - main_h[c]) / 2
        for nid in order.get(c, []):
            if by_id[nid].get("train_only"):
                continue
            y_top[nid] = cursor
            cursor += heights[nid] + ROW_GAP
        # train-only band: start all columns at the same train_top
        tcursor = train_top
        for nid in order.get(c, []):
            if not by_id[nid].get("train_only"):
                continue
            y_top[nid] = tcursor
            tcursor += heights[nid] + ROW_GAP

    train_tallest = max(train_h.values()) if train_h else 0.0
    total_h = main_tallest + (band_gap + train_tallest if any_train else 0)
    canvas_w = PAD_L + ncols * BOX_W + (ncols - 1) * COL_GAP + PAD_R
    canvas_h = pad_top + total_h + PAD_BOT

    return {
        "by_id": by_id, "idx": idx, "col": col, "order": order, "ncols": ncols,
        "x": col_x, "y_top": y_top, "heights": heights, "desclines": desclines,
        "truncs": truncs, "canvas_w": canvas_w, "canvas_h": canvas_h,
        "pad_top": pad_top, "fb_edges": fb_edges, "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

def render_arch_svg(arch: dict[str, Any]) -> tuple[str, int, int, list[str]]:
    L = _layout(arch)
    by_id, col, x, y_top, heights = L["by_id"], L["col"], L["x"], L["y_top"], L["heights"]
    W, H = L["canvas_w"], L["canvas_h"]
    edges = arch.get("edges", [])
    parts: list[str] = []

    def cx_right(nid): return x[col[nid]] + BOX_W
    def cx_left(nid): return x[col[nid]]

    # ports: spread outbound on the right edge, inbound on the left edge
    out_edges: dict[str, list] = {}
    in_edges: dict[str, list] = {}
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a in by_id and e.get("kind") in ("dataflow", "loss", "skip") and col.get(b, 0) > col.get(a, 0):
            out_edges.setdefault(a, []).append(e)
            in_edges.setdefault(b, []).append(e)

    def out_port(nid, e):
        lst = sorted(out_edges.get(nid, []), key=lambda ee: y_top.get(ee["to"], 0))
        k = lst.index(e); K = len(lst)
        return cx_right(nid), y_top[nid] + heights[nid] * (k + 1) / (K + 1)

    def in_port(nid, e):
        lst = sorted(in_edges.get(nid, []), key=lambda ee: y_top.get(ee["from"], 0))
        k = lst.index(e); K = len(lst)
        return cx_left(nid), y_top[nid] + heights[nid] * (k + 1) / (K + 1)

    # ---- group banners (drawn first, behind everything) ----
    groups = arch.get("groups", [])
    # the banner covers train-only *blocks*, but NOT loss nodes (losses live in the
    # right-most column as the natural endpoint; boxing them makes the band span the whole
    # figure). The reference figure bands the auxiliary branch, not the loss column.
    train_ids = [n["id"] for n in arch.get("nodes", [])
                 if n.get("train_only") and by_id[n["id"]].get("role") != "loss"]
    banner_specs = []
    if groups:
        for g in groups:
            members = g.get("members") or train_ids
            members = [m for m in members if m in by_id and by_id[m].get("role") != "loss"]
            if members:
                banner_specs.append((g.get("label", "TRAINING"), members))
    elif train_ids:
        banner_specs.append(("ONLY DURING TRAINING", train_ids))

    for label, members in banner_specs:
        x0 = min(x[col[m]] for m in members) - 12
        x1 = max(x[col[m]] + BOX_W for m in members) + 12
        y0 = min(y_top[m] for m in members) - 12
        y1 = max(y_top[m] + heights[m] for m in members) + 12
        # the pill sits ABOVE the band so it never overlaps the first member box
        pill_w = est_text_width(label, 11, bold=True) + 24
        pill_y = y0 - 22
        parts.append(
            f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{x1-x0:.0f}" height="{y1-y0:.0f}" rx="14" '
            f'fill="#e06c9a" fill-opacity="0.06" stroke="#e06c9a" stroke-opacity="0.45" '
            f'stroke-dasharray="6 5"/>'
        )
        parts.append(
            f'<rect x="{x0:.0f}" y="{pill_y:.0f}" width="{pill_w:.0f}" height="19" rx="9.5" fill="#e06c9a"/>'
            f'<text x="{x0+pill_w/2:.0f}" y="{pill_y+13:.0f}" fill="#1a0410" font-size="11" '
            f'font-weight="700" text-anchor="middle" letter-spacing="0.5">{_esc(label)}</text>'
        )

    # ---- forward edges (bezier) ----
    edge_labels = []
    for e in edges:
        a, b = e.get("from"), e.get("to")
        if a not in by_id or b not in by_id:
            continue
        kind = e.get("kind", "dataflow")
        is_forward = col[b] > col[a] and kind in ("dataflow", "loss", "skip")
        if not is_forward:
            continue
        st = EDGE_STYLE.get(kind, EDGE_STYLE["dataflow"])
        xr, yr = out_port(a, e)
        xl, yl = in_port(b, e)
        ch = 0.45 * COL_GAP
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        parts.append(
            f'<path d="M {xr:.0f} {yr:.0f} C {xr+ch:.0f} {yr:.0f} {xl-ch:.0f} {yl:.0f} {xl:.0f} {yl:.0f}" '
            f'fill="none" stroke="{st["color"]}" stroke-width="{st["width"]}"{dash} marker-end="url(#{st["marker"]})"/>'
        )
        if e.get("label"):
            mx, my = (xr + xl) / 2, (yr + yl) / 2 - 4
            edge_labels.append((mx, my, e["label"], st["color"]))

    # ---- feedback / backward edges (top channel) ----
    fb = [e for e in edges if e.get("from") in by_id and e.get("to") in by_id and (
        e.get("kind") == "feedback" or (e.get("kind") == "dataflow" and col[e["to"]] <= col[e["from"]]))]
    fb = sorted(fb, key=lambda e: x[col[e["from"]]])
    chy_base = L["pad_top"] - 14
    if len(fb) > 4:
        # too many — collapse to a badge near the most common source
        for e in fb[:0]:
            pass
    for lane, e in enumerate(fb[:4]):
        a, b = e["from"], e["to"]
        st = EDGE_STYLE["feedback"]
        xa = x[col[a]] + BOX_W / 2
        xb = x[col[b]] + BOX_W / 2
        ya = y_top[a]
        yb = y_top[b]
        chy = chy_base - lane * FEEDBACK_CH
        parts.append(
            f'<path d="M {xa:.0f} {ya:.0f} L {xa:.0f} {chy:.0f} L {xb:.0f} {chy:.0f} L {xb:.0f} {yb:.0f}" '
            f'fill="none" stroke="{st["color"]}" stroke-width="{st["width"]}" stroke-dasharray="{st["dash"]}" '
            f'marker-end="url(#ar-feedback)"/>'
        )
        if e.get("label"):
            edge_labels.append(((xa + xb) / 2, chy - 4, e["label"], st["color"]))

    # ---- boxes ----
    for n in arch.get("nodes", []):
        nid = n["id"]
        if nid not in by_id:
            continue
        bx, by = x[col[nid]], y_top[nid]
        h = heights[nid]
        fill, stroke = _node_color(n.get("role", "other"))
        low = (n.get("confidence", 1.0) or 1.0) < 0.6
        opacity = "0.62" if (low or n.get("train_only")) else "1"
        parts.append(f'<g class="stage" data-detail="{_esc(n.get("detail",""))}" opacity="{opacity}">')
        dash = ' stroke-dasharray="5 4"' if n.get("train_only") else ""
        parts.append(
            f'<rect x="{bx:.0f}" y="{by:.0f}" width="{BOX_W}" height="{h}" rx="12" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        parts.append(f'<rect x="{bx:.0f}" y="{by:.0f}" width="6" height="{h}" rx="3" fill="{stroke}"/>')
        ty = by + 19
        qmark = ' <tspan fill="#ffce3f">?</tspan>' if low else ""
        parts.append(
            f'<text x="{bx+14:.0f}" y="{ty:.0f}" fill="#ffffff" font-size="13" font-weight="650">'
            f'{_esc(_clip(n.get("title", nid), 30))}{qmark}</text>'
        )
        ly = ty + 17
        for line in L["desclines"][nid]:
            parts.append(f'<text x="{bx+14:.0f}" y="{ly:.0f}" fill="#c2c6cc" font-size="11.5">{_esc(line)}</text>')
            ly += LINE_H
        shp = _fmt_shape_arch(n.get("shape"))
        if shp:
            parts.append(
                f'<text x="{bx+14:.0f}" y="{by+h-10:.0f}" fill="#8a93a6" font-size="11" '
                f'font-family="ui-monospace,Menlo,monospace">{_esc(shp)}</text>'
            )
        parts.append("</g>")

    # ---- edge labels (on top) ----
    for mx, my, lbl, color in edge_labels:
        parts.append(
            f'<text x="{mx:.0f}" y="{my:.0f}" fill="{color}" font-size="10.5" text-anchor="middle" '
            f'style="paint-order:stroke;stroke:#0b0d12;stroke-width:3px">{_esc(_clip(str(lbl),28))}</text>'
        )

    # markers
    defs = "<defs>"
    for mid, color in [("ar-data", "#5b6472"), ("ar-loss", "#e06c9a"),
                       ("ar-feedback", "#c9a227"), ("ar-skip", "#ffe08a")]:
        defs += (f'<marker id="{mid}" markerWidth="9" markerHeight="9" refX="7" refY="3" '
                 f'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L7,3 L0,6 Z" fill="{color}"/></marker>')
    defs += "</defs>"

    svg = (f'<svg id="flow" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
           f'xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
           + defs + "".join(parts) + "</svg>")
    return svg, int(W), int(H), L["warnings"]


def _legend(arch: dict[str, Any]) -> str:
    used_roles, seen = [], set()
    for n in arch.get("nodes", []):
        r = n.get("role", "other")
        if r not in seen:
            seen.add(r); used_roles.append(r)
    rows = []
    for r in used_roles:
        _f, stroke = _node_color(r)
        rows.append(f'<div class="row"><span class="sw" style="background:{stroke}"></span>{_esc(r)}</div>')
    used_kinds = []
    for e in arch.get("edges", []):
        if e.get("kind") not in used_kinds:
            used_kinds.append(e.get("kind"))
    names = {"dataflow": "data flow", "loss": "loss flow", "feedback": "feedback / recurrent", "skip": "skip / reuse"}
    for k in used_kinds:
        st = EDGE_STYLE.get(k, EDGE_STYLE["dataflow"])
        rows.append(f'<div class="row"><span class="sw" style="background:{st["color"]}"></span>{_esc(names.get(k,k))}</div>')
    return "".join(rows)


def render_arch_html(arch: dict[str, Any], out_path: str, title: str | None = None) -> tuple[str, list[str]]:
    svg, _w, _h, warnings = render_arch_svg(arch)
    title = title or arch.get("model_name", "model")
    conf = arch.get("global_confidence")
    sub = (f'{arch.get("arch_family","")} · {len(arch.get("nodes", []))} blocks · {len(arch.get("edges", []))} edges'
           + (f' · confidence {round(conf*100)}%' if isinstance(conf, (int, float)) else "")
           + ' · click a box for detail')
    html_out = (_TEMPLATE
                .replace("__TITLE__", _esc(title))
                .replace("__SUB__", _esc(sub))
                .replace("__LEGEND__", _legend(arch))
                .replace("__SVG__", svg))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_out, encoding="utf-8")
    return str(p), warnings
