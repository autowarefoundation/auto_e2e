# ruff: noqa
"""Minimal SVG helper for block diagrams (no external deps)."""
from __future__ import annotations

import html
from dataclasses import dataclass, field

FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, monospace"

STYLES = {
    "input": dict(fill="#E3EEF9", stroke="#3F6FAE", dash=None),
    "frozen": dict(fill="#EEF0F4", stroke="#6E7B8E", dash=None),
    "trainable": dict(fill="#E2F3E4", stroke="#2E8B57", dash=None),
    "dormant": dict(fill="#FAFAFA", stroke="#9A9A9A", dash="5,3"),
    "output": dict(fill="#FBE5E3", stroke="#B8443F", dash=None),
    "loss": dict(fill="#FCEBF1", stroke="#B1407A", dash=None),
    "aux": dict(fill="#FCF4DB", stroke="#B8860B", dash=None),
    "plain": dict(fill="#FFFFFF", stroke="#444444", dash=None),
    "group": dict(fill="none", stroke="#8A8A8A", dash="6,4"),
    "note": dict(fill="#FFFDF2", stroke="#C9B458", dash=None),
}


@dataclass
class SVG:
    width: float
    height: float
    parts: list[str] = field(default_factory=list)
    font_size: float = 11.0

    def header(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'font-family="{FONT}" font-size="{self.font_size}">\n'
            '<defs>\n'
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/></marker>\n'
            '<marker id="arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8A8A8A"/></marker>\n'
            '<marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#B8443F"/></marker>\n'
            '</defs>\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white"/>\n'
        )

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, style="plain", rx=6, opacity=1.0, extra=""):
        st = STYLES[style]
        dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.2"'
            f'{dash} fill-opacity="{opacity}" {extra}/>'
        )

    def text(self, x, y, s, size=None, weight="normal", anchor="start",
             color="#222", family=None, italic=False, rotate=None):
        size = size or self.font_size
        fam = family or FONT
        style = ' font-style="italic"' if italic else ""
        rot = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.add(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{color}" font-family="{fam}"{style}{rot}>'
            f'{html.escape(s)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), style="plain", title_size=None,
            line_size=None, rx=6, mono_lines=(), badge=None):
        self.rect(x, y, w, h, style, rx=rx)
        ts = title_size or self.font_size
        ls = line_size or (self.font_size - 1.5)
        cy = y + ts + 4
        self.text(x + w / 2, cy, title, size=ts, weight="bold", anchor="middle")
        cy += ls + 3
        for ln in lines:
            fam = MONO if ln in mono_lines else None
            self.text(x + w / 2, cy, ln, size=ls, anchor="middle", family=fam, color="#333")
            cy += ls + 2.5
        if badge:
            bw = 7 * len(badge) + 8
            self.add(
                f'<rect x="{x + w - bw - 3}" y="{y - 8}" width="{bw}" height="14" rx="7" '
                f'fill="#444" />'
            )
            self.text(x + w - bw / 2 - 3, y + 2.5, badge, size=8.5, anchor="middle",
                      color="white", weight="bold")

    def arrow(self, x1, y1, x2, y2, label=None, color="#333", marker="arrow",
              dash=None, label_dy=-4, width=1.3, label_size=None, elbow=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if elbow is None:
            path = f"M {x1} {y1} L {x2} {y2}"
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        elif elbow == "h":  # horizontal first then vertical
            path = f"M {x1} {y1} L {x2} {y1} L {x2} {y2}"
            lx, ly = (x1 + x2) / 2, y1
        elif elbow == "v":  # vertical first then horizontal
            path = f"M {x1} {y1} L {x1} {y2} L {x2} {y2}"
            lx, ly = x1, (y1 + y2) / 2
        elif isinstance(elbow, (int, float)):  # elbow at x = elbow
            path = f"M {x1} {y1} L {elbow} {y1} L {elbow} {y2} L {x2} {y2}"
            lx, ly = elbow, (y1 + y2) / 2
        else:
            path = elbow
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
        self.add(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"'
            f'{d} marker-end="url(#{marker})"/>'
        )
        if label:
            self.text(lx, ly + label_dy, label, size=label_size or (self.font_size - 2),
                      anchor="middle", color=color)

    def line(self, x1, y1, x2, y2, color="#333", dash=None, width=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')

    def legend(self, x, y, items, size=None):
        size = size or (self.font_size - 1)
        cx = x
        for style, label in items:
            st = STYLES[style]
            dash = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
            self.add(
                f'<rect x="{cx}" y="{y - 9}" width="14" height="10" fill="{st["fill"]}" '
                f'stroke="{st["stroke"]}"{dash}/>'
            )
            self.text(cx + 18, y, label, size=size)
            cx += 18 + 6.2 * len(label) + 14

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.header())
            f.write("\n".join(self.parts))
            f.write("\n</svg>\n")
