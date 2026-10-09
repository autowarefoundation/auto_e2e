# ruff: noqa
from pathlib import Path
import re, subprocess, sys
ROOT=Path('/tmp/autoe2e_paper/vivlio')
files={k:(ROOT/f'paper_{k}.md').read_text() for k in ('ja','en')}
errors=[]; evidence=[]
def require(cond,msg,ev=None):
    if not cond: errors.append(msg)
    elif ev: evidence.append(ev)
# Author and affiliation
for k,t in files.items():
    require('Ryota Yamada' in t and 'Amazon Web Services, Inc.' in t, f'{k}: author/affiliation missing', f'{k}: author and affiliation present')
# Abstract length
m=re.search(r'## Abstract:\{\.abstract lang="en"\}\n\n(.*?)\n\n## Keywords',files['en'],re.S)
wc=len(re.findall(r"\b[\w.-]+\b",m.group(1))) if m else -1
require(180<=wc<=230,f'English abstract length {wc} outside 180-230',f'English abstract = {wc} words')
# Architecture positioning and temporal distinctions
require('UniAD-inspired but narrower' in files['en'], 'UniAD-inspired positioning missing', 'UniAD-inspired, narrower scope stated')
require('3.5 s' in files['en'] and '6.4 s' in files['en'], 'temporal spans missing', '3.5 s camera and 6.4 s egomotion spans present')
require('1024' in files['en'] and 'current frame only' in files['en'], 'front branch scope missing', '1024-pixel branch limited to current frame')
require('has not been run' in files['en'] and 'not a map ablation' in files['en'], 'ablation limitation missing', 'matched ablation explicitly unrun; Test not called ablation')
# Exact metrics
expected=[
'0.1472','0.3729','0.7449','1.9405','0.1347','0.3886','0.8035','2.0952','0.1951','0.4723','0.9147','2.2951','0.2030','0.5018','0.9503','2.2919',
'0.2850','0.9239','2.0196','5.5645','0.2831','1.0101','2.2127','5.9509','0.3779','1.1331','2.4226','6.3885','0.4066','1.1935','2.4508','6.2272',
'0.1463','0.4121','0.8263','2.1042','0.1680','0.4923','0.9831','2.4699','0.1602','0.3979','0.7924','2.1120','0.1916','0.4817','0.9181','2.2840',
'0.3107','1.0453','2.2199','5.9422','0.3754','1.2512','2.6242','6.8964','0.3134','0.9727','2.1621','6.1751','0.3899','1.1514','2.3895','6.4266',
'0.9844','1.4025','1.0220','1.5659','1.1911','1.6633','1.2082','1.6448','1.2044','1.4051','1.4730','1.6598','1.0584','1.5496','1.2167','1.6287',
'1.2578','3.8450','0.9912','1.2668','3.8982','0.9980','2.6326','7.5567','0.9991','2.7548','7.8574','0.9990',
'0.3959','0.4236','0.0608','0.0763','44.7857','45.5856','0.8725','0.8982','0.7758','0.7988','0.4442','0.4139','0.7542','0.7466','0.2458','0.2534',
'11,035','23,690','551,750','706,240','1,184,500','1,516,160','78.125%','79,906,522','1,448,582']
for k,t in files.items():
    missing=[x for x in expected if x not in t]
    require(not missing,f'{k}: missing exact values {missing[:10]}',f'{k}: all {len(expected)} audited result/config values present')
# External tables must not claim 6.4s
for table_id in ('tab-val-ade','tab-test-ade'):
    for k,t in files.items():
        m=re.search(rf'<figure[^>]+id="{table_id}".*?</figure>',t,re.S)
        require(m is not None and not re.search(r'6\.4\s*s',m.group(0)),f'{k}: external table {table_id} has a 6.4-second metric or is missing',f'{k}: {table_id} limited to 1/2/3/5s')
# Reference integrity
for k,t in files.items():
    refs=set(re.findall(r'<div class="reference" id="([^"]+)"',t))
    links=set(re.findall(r'<a class="cite" href="#([^"]+)"',t))
    require(links<=refs,f'{k}: undefined citations {sorted(links-refs)}',f'{k}: {len(links)} cited works all defined ({len(refs)} references)')
    imgs=re.findall(r'<img src="([^"]+)"',t)
    missing=[x for x in imgs if not (ROOT/x).is_file()]
    require(not missing,f'{k}: missing images {missing}',f'{k}: all {len(imgs)} referenced figures exist')
# Privacy / prohibited strings
for k,t in files.items():
    bad=[]
    if 's3://' in t: bad.append('s3 URI')
    if re.search(r'\b\d{12}\b',t): bad.append('12-digit account')
    if re.search(r'[a-z0-9.-]+\.internal\b',t): bad.append('internal hostname')
    require(not bad,f'{k}: private provenance leaked {bad}',f'{k}: no S3 URI/account ID/internal hostname')
# PDFs and LaTeX
pdfs=[ROOT/'autoe2e_paper_ja.pdf',ROOT/'autoe2e_paper_en.pdf',ROOT/'paper_en.pdf']
for p in pdfs:
    require(p.is_file() and p.stat().st_size>500_000,f'PDF missing/small: {p}',f'{p.name}: {p.stat().st_size:,} bytes')
require((ROOT/'paper_en.tex').is_file() and '\\begin{document}' in (ROOT/'paper_en.tex').read_text(),'complete LaTeX missing','standalone LaTeX has document environment')
# PDF text author check
for lang in ('ja','en'):
    p=ROOT/f'autoe2e_paper_{lang}.pdf'; out=Path(f'/tmp/audit_{lang}.txt')
    subprocess.run(['pdftotext',str(p),str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    txt=out.read_text(errors='ignore')
    require('Ryota Yamada' in txt and 'Amazon Web Services' in txt,f'{lang} PDF author missing',f'{lang} PDF contains final author/affiliation')
print('AUDIT EVIDENCE')
for e in evidence: print('PASS',e)
if errors:
    print('AUDIT FAILURES')
    for e in errors: print('FAIL',e)
    sys.exit(1)
print(f'AUDIT PASSED: {len(evidence)} checks')
