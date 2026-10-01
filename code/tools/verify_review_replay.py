#!/usr/bin/env python3
"""Load every archived model and freshly reevaluate prespecified representatives."""
from __future__ import annotations
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/compare_fem_rfm/src'))
import numpy as np
from compare_fem_rfm.experiment import _evaluate_feature_solution
from compare_fem_rfm.features import FeatureSet
from compare_fem_rfm.fem import FEMModel,evaluate_fem_model
from compare_fem_rfm.metrics import evaluate_fields
from compare_fem_rfm.problems import CASES
from compare_fem_rfm.quadrature import composite_gauss


def main():
    rows=[]
    base=ROOT/'data/raw/review_replay_20261001'
    for method in ('rfm','cga','fem'):
        with (base/f'{method}_raw.csv').open(newline='') as f:
            rows+=list(csv.DictReader(f))
    assert len(rows)==466
    for row in rows:
        path=ROOT/row['model_path']
        assert sha256(path.read_bytes()).hexdigest()==row['model_sha256']
        with np.load(path,allow_pickle=False) as model:
            assert all(np.isfinite(model[name]).all() for name in model.files),str(path)
            if row['method']!='fem':
                assert len(model['coefficients'])==int(row['dof'])
                assert model['w'].shape==(int(row['dof']),CASES[row['case_id']].dim)
    chosen={}
    for row in rows:
        if row['method']=='rfm' and row['seed']!='201':
            continue
        key=row['case_id'],row['method'],row['variant']
        if key not in chosen or int(row['dof'])>int(chosen[key]['dof']):
            chosen[key]=row
    representatives=list(chosen.values())
    representatives += [r for r in rows if r['case_id']=='C3' and r['method']=='rfm'
                        and r['seed'] in {'206','207','209'} and r['dof']=='512']
    checks=[]
    for row in representatives:
        spec=CASES[row['case_id']]
        cells,order,dim=map(int,re.fullmatch(r'CG(\d+)x(\d+)-d(\d+)',row['evaluation_rule']).groups())
        assert dim==spec.dim
        rule=composite_gauss(dim,cells,order)
        assert rule.hash==row['evaluation_hash']
        with np.load(ROOT/row['model_path']) as saved:
            if row['method']=='fem':
                model=FEMModel(int(saved['dim']),int(saved['degree']),
                               int(saved['n_elements_axis']),saved['coefficients'])
                u,grad=evaluate_fem_model(model,rule.points,batch_size=4096)
                metrics=evaluate_fields(spec,rule.points,rule.weights,u,grad)
            else:
                features=FeatureSet(saved['w'],saved['b'],int(saved['k']),saved['scales'],saved['centers'])
                metrics=_evaluate_feature_solution(spec,features,saved['coefficients'],rule,batch_size=4096)
        errors={}
        for metric in ('energy_gap','natural_error','v_error','l2_error'):
            if metrics[metric] is None:
                assert row[metric] in ('','NA')
                continue
            expected=float(row[metric]); observed=metrics[metric]
            assert np.isclose(expected,observed,rtol=1e-10,atol=1e-13),(row['model_path'],metric,expected,observed)
            errors[metric]=abs(expected-observed)
        checks.append(dict(case_id=row['case_id'],method=row['method'],variant=row['variant'],
                           seed=row['seed'],dof=int(row['dof']),model_path=row['model_path'],
                           metric_absolute_differences=errors,passed=True))
        print('Fresh evaluation:',row['model_path'],flush=True)
    report=dict(status='PASS',all_models_loaded=len(rows),representatives_reevaluated=len(checks),
                selection='largest recorded state for each case/method/FEM degree; RFM seed201 plus C3 seeds206,207,209 at N512',
                scope='fresh loading and reevaluation of saved coefficients, not independent PDE solves',checks=checks)
    target=ROOT/'review/review_replay_reevaluation.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':
    main()
