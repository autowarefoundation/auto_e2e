#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p figures_pdf
for f in figures/*.svg; do
  b=$(basename "$f" .svg)
  rsvg-convert -f pdf "$f" -o "figures_pdf/$b.pdf"
done
npx --yes @vivliostyle/vfm@2.7.2 paper_en.md --math-renderer mathml > "$ROOT/paper_en.html"
pandoc "$ROOT/paper_en.html" -f html -t latex -s --top-level-division=section \
  -V geometry:a4paper -V geometry:margin=18mm -V fontsize=10pt -o paper_en.tex
python3 - <<'PY'
from pathlib import Path
import re
p=Path('paper_en.tex'); t=p.read_text()
t=re.sub(r'\\pandocbounded\{\\includesvg\[keepaspectratio\]\{figures/([^}]+)\.svg\}\}',
         r'\\pandocbounded{\\includegraphics[keepaspectratio,width=\\linewidth]{figures_pdf/\1.pdf}}', t)
t=re.sub(r'(figures_pdf/[^}\n]+)\\_([^}\n]+\.pdf)', lambda m:m.group(1)+'_'+m.group(2), t)
t=t.replace('\\begin{longtable}', '\\begingroup\\scriptsize\n\\begin{longtable}')
t=t.replace('\\end{longtable}', '\\end{longtable}\n\\endgroup')
t=t.replace('β', '$\\beta$').replace('◯', '$\\circ$')
# Pandoc converts reference divs to bibitems but omits the environment.
first='\\bibitem[\\citeproctext]{ref-uniad}'
if first in t:
    t=t.replace(first, '\\begin{thebibliography}{99}\n\\bibitem{ref-uniad}', 1)
t=t.replace('\\bibitem[\\citeproctext]{', '\\bibitem{')
append='\\subsection{Appendix}\\label{appendix}'
if append in t and '\\end{thebibliography}' not in t:
    t=t.replace(append, '\\end{thebibliography}\n\\clearpage\n\\appendix\n'+append, 1)
p.write_text(t)
PY
tectonic -X compile paper_en.tex --outdir .
