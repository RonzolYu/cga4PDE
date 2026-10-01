#!/usr/bin/env python3
"""Rebuild the RFM seed audit and main-text denominator table; verify existing plots.

No PDE solve is performed.  Standard Python and pypdf are required.  Run from
any directory; the package root is inferred from this script's location.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/compare_fem_rfm/src'))
from compare_fem_rfm.quality import metric_valid, invalid_reason

METRICS = ('energy_gap', 'natural_error', 'v_error')
METRIC_MAP = {'energy_gap': 'energy_gap', 'natural_error': 'relative_sobolev', 'v_error': 'v_distance'}

def read(path):
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def num(v):
    try: x=float(v)
    except (ValueError, TypeError): return None
    return x if math.isfinite(x) else None

def quantile(values, p):
    v=sorted(values); z=(len(v)-1)*p; j=int(z)
    return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(z-j)

def check_same(x,y,label):
    assert math.isclose(float(x),float(y),rel_tol=2e-12,abs_tol=1e-14), (label,x,y)

def pdf_paths(path, rgb=(215/255,48/255,39/255)):
    """Read red RFM line and band coordinates without rasterization or model fitting."""
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    pdf=PdfReader(path); operations=ContentStream(pdf.pages[0].get_contents(),pdf).operations
    color=(0.,0.,0.); fill=color; points=[]; stroke=[]; bands=[]; stack=[]
    red=rgb
    def is_red(c): return max(abs(a-b) for a,b in zip(c,red))<1e-6
    for args,op in operations:
        if op==b'q': stack.append((color,fill))
        elif op==b'Q' and stack: color,fill=stack.pop()
        elif op==b'RG': color=tuple(float(a) for a in args)
        elif op==b'rg': fill=tuple(float(a) for a in args)
        elif op==b'G': color=(float(args[0]),)*3
        elif op==b'g': fill=(float(args[0]),)*3
        elif op==b'm': points=[tuple(float(a) for a in args)]
        elif op==b'l': points.append(tuple(float(a) for a in args))
        elif op in {b'S',b's',b'f',b'f*',b'B',b'B*',b'n'}:
            if len(points)>1:
                if op in {b'S',b's'} and is_red(color): stroke.append(points[:])
                if op in {b'f',b'f*'} and is_red(fill): bands.append(points[:])
            points=[]
    if not stroke:
        raise AssertionError(f'No curve with RGB {rgb} in {path}')
    return max(stroke,key=len), max(bands,key=len) if bands else []

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args()
    root=args.root.resolve(); target=root/'tex'; review=root/'review'
    paths={name:root/rel for name,rel in {
        'raw':'result/experiments/rfm_multiseed_raw.csv',
        'summary':'result/experiments/rfm_multiseed_summary.csv',
        'actual':'result/baselines/baseline_actual_points.csv',
        'terminal':'result/baselines/baseline_terminal.csv',
        'config':'config/plots.json'}.items()}
    raw=read(paths['raw']); old_summary=read(paths['summary']); actual=read(paths['actual']); terminal=read(paths['terminal']); cfg=json.loads(paths['config'].read_text())
    groups=defaultdict(list)
    for r in raw: groups[(r['case_id'],int(r['dof']))].append(r)
    assert len({(r['case_id'],r['dof'],r['seed']) for r in raw})==len(raw)
    old={(r['case_id'],int(r['dof'])):r for r in old_summary}
    actual_keys={(r['case_id'],int(r['dof']),r['metric']) for r in actual if r['method']=='RFM'}
    ledger=[]; stats=[]; verification=[]
    for key, group in sorted(groups.items()):
        case,n=key; prescribed=10; observed=len(group); succeeded=[r for r in group if r['solver_success']=='True']
        assert observed==int(old[key]['seed_count'])
        assert len(succeeded)==int(old[key]['success_count'])
        for r in sorted(group,key=lambda a:int(a['seed'])):
            row={'case_id':case,'N':n,'seed':r['seed'],'prescribed_seeds':10,'observed_seeds_at_width':observed,
                 'solver_success':r['solver_success'],'failure_reason':'none' if r['solver_success']=='True' else 'recorded solver_success=False; '+r['solver_message'],
                 'solver_message':r['solver_message']}
            for m in METRICS:
                applicable=(m!='v_error' or case in {'C4','C5'})
                value=num(r[m]); valid=applicable and value is not None and value>=0
                participates=applicable and metric_valid(r, m)
                displayed=participates and (case,n,METRIC_MAP[m]) in actual_keys and n<=int(cfg['baseline_caps'][case])
                row[m]=r[m]; row[m+'_applicable']=applicable; row[m+'_finite_nonnegative']=valid if applicable else 'not applicable'
                row[m+'_included_in_summary']=participates; row[m+'_included_in_main_figure']=displayed
                row[m+'_invalid_reason']=invalid_reason(r,m)
            ledger.append(row)
        sr={'case_id':case,'N':n,'prescribed_seeds':10,'observed_seeds':observed,'successes':len(succeeded),'failures':observed-len(succeeded),'ten_seed_group':observed==10}
        for m in METRICS:
            values=[num(r[m]) for r in group if metric_valid(r,m)]
            sr[m+'_sample_count']=len(values)
            assert len(values)==int(old[key][m+'_sample_count'])
            for p,suffix in [(.25,'q1'),(.5,'median'),(.75,'q3')]:
                value=quantile(values,p) if values else ''
                sr[m+'_'+suffix]=value
                if values: check_same(value,old[key][m+'_'+suffix],str((key,m,suffix)))
                if (case,n,METRIC_MAP[m]) in actual_keys:
                    a=next(r for r in actual if r['method']=='RFM' and r['case_id']==case and int(r['dof'])==n and r['metric']==METRIC_MAP[m])
                    check_same(value,a['value' if suffix=='median' else suffix],'actual '+str((key,m,suffix)))
        stats.append(sr)
    write(review/'rfm_state_audit.csv',ledger); write(review/'rfm_recomputed_summary.csv',stats)
    for t in terminal:
        case,n=t['case_id'],int(t['dof']); s=next(r for r in stats if r['case_id']==case and r['N']==n)
        for suffix in ['median','q1','q3']: check_same(t['rfm_'+suffix],s['natural_error_'+suffix],case+' endpoint '+suffix)
        ratio=float(t['rfm_median'])/float(t['cga']); femratio=float(t['fem_p3'])/float(t['cga'])
        check_same(t['rfm_over_cga'],ratio,case+' RFM/CGA'); check_same(t['fem_p3_over_cga'],femratio,case+' FEM/CGA')
        assert int(t['rfm_success_count'])==s['natural_error_sample_count']
        verification.append({'case_id':case,'N':n,'RFM_n':s['natural_error_sample_count'],'median':s['natural_error_median'],'q1':s['natural_error_q1'],'q3':s['natural_error_q3'],'RFM_CGA':ratio,'FEM_P3_CGA':femratio})
    write(review/'baseline_endpoint_verification.csv',verification)
    # The manuscript PDFs are vector figures. Check the RFM median and IQR paths
    # against the current CSV values after the plot's log-affine coordinate map.
    figure_checks=[]
    for case in ['C1','C2','C3','C4','C5']:
        metrics=['energy_gap','relative_sobolev']+(['v_distance'] if case in {'C4','C5'} else [])
        for metric in metrics:
            suffix=metric if metric!='relative_sobolev' else ('relative_h1_error' if case in {'C1','C2','C3'} else 'relative_w1p_error')
            path=target/'figures/baselines'/f'{case}_{suffix}.pdf'
            data=sorted([r for r in actual if r['case_id']==case and r['method']=='RFM' and r['metric']==metric and int(r['dof'])<=int(cfg['baseline_caps'][case])],key=lambda a:int(a['dof']))
            line,band=pdf_paths(path); assert len(line)==len(data),(path,len(line),len(data))
            lx=[math.log(int(r['dof'])) for r in data]; ly=[math.log(float(r['value'])) for r in data]
            bx=(line[-1][0]-line[0][0])/(lx[-1]-lx[0]); ax=line[0][0]-bx*lx[0]
            by=(line[-1][1]-line[0][1])/(ly[-1]-ly[0]); ay=line[0][1]-by*ly[0]
            error=max(max(abs(x-(ax+bx*u)),abs(y-(ay+by*v))) for (x,y),u,v in zip(line,lx,ly))
            expected_band=[(ax+bx*u,ay+by*math.log(float(r[q]))) for u,r in zip(lx,data) for q in ['q1','q3']]
            band_error=max(min(max(abs(x-ex),abs(y-ey)) for ex,ey in expected_band) for x,y in band)
            assert max(error,band_error)<1e-4,(str(path),error,band_error)
            all_errors=[]
            for method,color in [('CGA',(118/255,42/255,131/255)),('FEM P1',(69/255,117/255,180/255)),('FEM P3',(26/255,152/255,80/255))]:
                other,_=pdf_paths(path,color)
                values=sorted([r for r in actual if r['case_id']==case and r['method']==method and r['metric']==metric and int(r['dof'])<=int(cfg['baseline_caps'][case]) and float(r['value'])>0],key=lambda a:int(a['dof']))
                expected=[(ax+bx*math.log(int(r['dof'])),ay+by*math.log(float(r['value']))) for r in values]
                matched=[]
                for x,y in other:
                    nearest=min(expected,key=lambda q:abs(q[0]-x))
                    if abs(nearest[0]-x)<1e-4:
                        matched.append(max(abs(x-nearest[0]),abs(y-nearest[1])))
                    else:
                        # Matplotlib replaces off-page path endpoints by a page-edge intersection.
                        from pypdf import PdfReader
                        page_width=float(PdfReader(path).pages[0].mediabox.width)
                        assert x<0 or x>page_width,(str(path),method,'unmatched interior vertex',x,y)
                assert len(matched)>=sum(int(r['dof'])>=8 for r in values),(str(path),method,'missing visible point')
                e=max(matched)
                assert e<1e-4,(str(path),method,e)
                all_errors.append(e)
            figure_checks.append({'figure':str(path.relative_to(target)),'other_methods_max_error_pdf_points':max(all_errors),'rfm_points':len(data),'median_max_error_pdf_points':error,'IQR_max_error_pdf_points':band_error,'sha256':digest(path)})
    write(review/'baseline_figure_verification.csv',figure_checks)
    lines=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{\revtext{RFM counts at the common endpoints. Each width has ten prescribed seeds. The Sobolev median/IQR sample additionally passes the metric-specific signed-value and one-percent quadrature checks.}}',r'\label{tab:rfm-fixed-denominators}',r'\begin{revision}',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Case and width & Prescribed & Solved & Failed & Valid sample \\',r'\midrule']
    for t in terminal:
        case,n=t['case_id'],int(t['dof'])
        s=next(r for r in stats if r['case_id']==case and r['N']==n)
        lines.append(f'{case}, $N={n}$ & 10 & {s["successes"]} & {s["failures"]} & {s["natural_error_sample_count"]} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{revision}',r'\end{table}']
    (target/'generated/rfm_fixed_denominators.tex').write_text('\n'.join(lines)+'\n')
    archived=raw+read(root/'data/derived/experiments/cga_baseline_raw.csv')+read(root/'data/derived/experiments/fem_baseline_raw.csv')
    for row in archived:
        model=(root/row['model_path']).resolve()
        assert model.is_relative_to(root) and model.is_file(), row['model_path']
        assert digest(model)==row['model_sha256'], row['model_path']
    evidence={'status':'PASS','raw_rows':len(raw),'raw_successes':sum(r['solver_success']=='True' for r in raw),'raw_failures':sum(r['solver_success']!='True' for r in raw),'summary_groups':len(stats),'figures_checked':len(figure_checks),'archived_models_checked':len(archived),'source_sha256':{str(v.relative_to(root)):digest(v) for v in paths.values()},'validity_meaning':'successful solve, finite nonnegative signed metric, and successive quadrature relative difference <= 0.01; same rule for every method'}
    (review/'baseline_evidence_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
