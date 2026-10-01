#!/usr/bin/env python3
"""Validate a complete new batch before replacing the primary comparison inputs."""
from __future__ import annotations
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/compare_fem_rfm/src'))
from compare_fem_rfm.quality import summarize_rfm, metric_valid


def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def key(row):
    return row['case_id'],row['method'],row['variant'],row['seed'],row['dof']


def main():
    batch=ROOT/'data/raw/review_replay_20261001'
    config=json.loads((ROOT/'config/review_replay.json').read_text())
    assert (batch/'protocol.json').read_bytes()==(ROOT/'config/review_replay.json').read_bytes()
    rows={method:read(batch/f'{method}_raw.csv') for method in ('rfm','cga','fem')}
    for method, records in rows.items():
        update=batch/f'evaluation_updates_{method}.csv'
        if update.exists():
            updates={key(r):r for r in read(update)}
            assert set(updates)<=set(map(key,records)), 'unknown refreshed state'
            records[:]=[updates.get(key(r),r) for r in records]
        assert len(set(map(key,records)))==len(records), f'duplicate {method} state'
        for row in records:
            model=(ROOT/row['model_path']).resolve()
            assert model.is_relative_to(batch/'models') and model.is_file(), row['model_path']
            assert sha256(model.read_bytes()).hexdigest()==row['model_sha256'], row['model_path']
    expected={(case,str(seed),str(n)) for case,widths in config['widths'].items()
              for seed in config['seeds'] for n in widths}
    assert {(r['case_id'],r['seed'],r['dof']) for r in rows['rfm']}==expected
    expected_cga={(case,str(n)) for case,widths in config['widths'].items()
                  for n in (widths if case!='C4' else [8,16,32,64,128,141])}
    assert {(r['case_id'],r['dof']) for r in rows['cga']}==expected_cga
    old_fem=read(ROOT/'data/raw/baseline/data/fem_raw.csv')
    assert {(r['case_id'],r['variant'],r['mesh_level']) for r in rows['fem']}=={
        (r['case_id'],r['variant'],r['mesh_level']) for r in old_fem}
    summaries=summarize_rfm(rows['rfm'],config['validity']['audit_relative_tolerance'])
    evaluator_files=[ROOT/'scripts/compare_fem_rfm/src/compare_fem_rfm'/name
                     for name in ('metrics.py','fem.py','features.py','experiment.py','quality.py')]
    evaluator_hashes={p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest()
                      for p in evaluator_files}
    evaluator_hash=sha256(json.dumps(evaluator_hashes,sort_keys=True).encode()).hexdigest()
    for records in rows.values():
        for row in records:
            row.update(evaluator_hash=evaluator_hash,source_hash=row['model_sha256'],
                       is_raw=True,is_interpolated=False)
    (batch/'evaluators.json').write_text(json.dumps(evaluator_hashes,indent=2)+'\n')
    derived=ROOT/'data/derived/experiments'
    historical=ROOT/'data/raw/historical_baseline_202609'
    if not historical.exists():
        historical.mkdir()
        for name in ('rfm_multiseed_raw.csv','rfm_multiseed_summary.csv','experiment_validation.json'):
            shutil.copy2(derived/name,historical/name)
        shutil.copytree(ROOT/'data/derived/baselines',historical/'derived_baselines')
        (historical/'README.md').write_text(
            '# Historical comparison inputs\n\nThese are the pre-revision inputs from upstream '
            'commit 679f95c. They are retained for provenance, not used by the revised comparison. '
            'Their model archive is incomplete and their statistics do not apply the new '
            'quadrature policy. The new batch is explicitly a coefficient replay, not a '
            'reconstruction of missing historical states.\n',encoding='utf-8')
    for method,records in rows.items():
        write(batch/f'{method}_raw.csv',records)
        filename='rfm_multiseed_raw.csv' if method=='rfm' else f'{method}_baseline_raw.csv'
        for base in (derived,ROOT/'result/experiments'):
            write(base/filename,records)
    for base in (derived,ROOT/'result/experiments'):
        write(base/'rfm_multiseed_summary.csv',summaries)
    validation=json.loads((derived/'experiment_validation.json').read_text())
    validation['schema_version']='cga-experiment-validation-v2'
    validation['comparison_batch']='review_replay_20261001'
    validation['rfm']={'rows':len(rows['rfm']),'seeds':config['seeds'],
                       'failures':sum(r['solver_success']!='True' for r in rows['rfm'])}
    validation['comparison_archive']={m:len(rs) for m,rs in rows.items()}
    validation['comparison_metric_valid_counts']={m:{metric:sum(metric_valid(r,metric) for r in rs)
        for metric in ('energy_gap','natural_error','v_error')} for m,rs in rows.items()}
    validation['validity_policy']=config['validity']
    validation['tests_expected']={'cga_refactor':102,'compare_fem_rfm':15}
    for base in (derived,ROOT/'result/experiments'):
        (base/'experiment_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (batch/'archive_validation.json').write_text(json.dumps({
        'passed':True,'batch':'review_replay_20261001','model_counts':validation['comparison_archive'],
        'metric_valid_counts':validation['comparison_metric_valid_counts'],
        'protocol_sha256':sha256((batch/'protocol.json').read_bytes()).hexdigest(),
        'all_model_hashes_verified':True},indent=2)+'\n')
    print(json.dumps(validation['comparison_metric_valid_counts'],indent=2))


if __name__=='__main__':
    main()
