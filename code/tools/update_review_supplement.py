#!/usr/bin/env python3
"""Write the supplement sample ledger directly from the validated primary batch."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
def main():
    validation=json.loads((ROOT/'data/derived/experiments/experiment_validation.json').read_text())
    assert validation['comparison_batch']=='review_replay_20261001'
    summaries=read(ROOT/'data/derived/experiments/rfm_multiseed_summary.csv')
    lines=['# S3. Samples, solve failures, and metric-specific statistics (2026-10-01)', '',
        'The revised comparison uses the fully archived `review_replay_20261001` batch,',
        'not the historical comparison inputs. CGA coefficients are refitted on fixed',
        'greedy prefixes; FEM and RFM coefficients are newly fitted under the frozen',
        '`code/config/review_replay.json` protocol. Original CGA training trajectories',
        'and auxiliary finite-range diagnostics remain separate.', '',
        f"The RFM file has {validation['rfm']['rows']} observations and "
        f"{validation['rfm']['failures']} recorded solve failures. Every width has ten "
        'prescribed and observed seeds, 201–210. Solve success and metric validity are distinct.', '',
        'An observation contributes to an energy, Sobolev, or V statistic only if its',
        'solve succeeds, its metric is finite and nonnegative, and its successive',
        'evaluation-quadrature relative difference is at most 0.01. This policy is',
        'the same for CGA, FEM, and RFM. Missing or failed audits exclude that metric.',
        'Energy gaps retain their sign; negative gaps are retained raw and excluded',
        'from energy summaries. An energy exclusion does not itself exclude a stable',
        'Sobolev or V observation. Quadrature agreement is not a rigorous enclosure.', '',
        'C1–C3 use relative H1 errors. C4–C5 use the componentwise gradient L4',
        'seminorm, not the relative full W1,4 norm. The V metric is unavailable for',
        'C1–C3 without affecting their other statistics. Auxiliary L2 and timing',
        'summaries use successful solves; they are not primary audited comparisons.', '',
        '| Case | Width | Prescribed | Solved | Failed | Valid energy | Valid Sobolev | Valid V |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append('| '+' | '.join(str(row[k]) for k in ('case_id','dof','seed_count',
            'success_count','failure_count','energy_gap_sample_count','natural_error_sample_count','v_error_sample_count'))+' |')
    lines += ['', 'Quartiles use linear interpolation at zero-based positions `(n-1)*p`,',
        'with p=0.25, 0.5, 0.75. Groups with fewer than ten valid observations are',
        'conditional samples; exclusions remain in the prescribed denominator.', '',
        'Primary raw CSVs are `code/data/derived/experiments/rfm_multiseed_raw.csv`,',
        '`cga_baseline_raw.csv`, and `fem_baseline_raw.csv`. Every row records a',
        'model path/hash, evaluation rule/hash, metric audits, and exclusion reasons.',
        'There are 320 RFM, 32 CGA-prefix, and 114 FEM coefficient models.',
        'The RFM summary, plotted actual points, endpoints, and interpolation brackets',
        'are in `code/result/experiments/` and `code/result/baselines/`.', '',
        '`code/tools/rebuild_baseline_evidence.py` checks stored statistics and all',
        'twelve vector baseline figures. This is not an independent PDE solve.',
        'Fresh loading and reevaluation checks are described in S4. Historical',
        'inputs remain in `code/data/raw/historical_baseline_202609/` and do not',
        'supply these revised samples.']
    text='\n'.join(lines)+'\n'
    for path in (ROOT.parent/'paper/sisc_cga/supplement/S3_failures_and_statistics.md',
                 ROOT/'supplement_records/S3_failures_and_statistics.md'):
        path.write_text(text)
    print('Updated S3 from validated metric-specific sample counts.')
if __name__=='__main__':
    main()
