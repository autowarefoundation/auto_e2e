# AutoE2E: Open-Loop Evaluation of a Map- and Oracle-Route-Conditioned Temporal BEV Planner

This source bundle accompanies the English and Japanese editions of the AutoE2E technical paper.

## Contents

- `paper_en.md` / `paper_ja.md`: Vivliostyle Flavored Markdown manuscripts
- `paper_en.tex`: generated standalone LaTeX source
- `theme/paper.css`: A4 two-column Vivliostyle theme
- `figures/`: publication figures in SVG
- `scripts/`: figure generation, parameter counting, and audit helpers
- `evidence/`: source-inspection and bibliography-verification notes

## Scope and caveats

The evaluated configuration is `bevformer_v2_t8_split_navigation_v5`. The KITScenes route is an oracle post-hoc route reconstructed from the logged whole-scene trajectory. The reported results are open-loop. The Camera-only Test population differs from Val and is not a causal Map/Route ablation. No closed-loop safety, intervention, or state-of-the-art claim is made.

## Build

Install the pinned dependencies from `package.json`, then run `npm run build:en` or `npm run build:ja` from the project root. The LaTeX source can be compiled with Tectonic after converting SVG figures to PDF.

## License

Apache License 2.0. See `LICENSE`.
