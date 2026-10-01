#!/usr/bin/env python3
"""Merge completed disjoint seed workers into the canonical model archive."""
import csv
from hashlib import sha256
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/raw/review_replay_20261001'
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
def key(row):
    return row['case_id'],int(row['seed']),int(row['dof'])
def main():
    rows={key(r):r for r in read(BASE/'rfm_raw.csv')}
    workers=[BASE/f'worker_{i}' for i in range(3)]
    pending=[]
    for worker in workers:
        assert (worker/'protocol.json').read_bytes()==(BASE/'protocol.json').read_bytes()
        records=read(worker/'rfm_raw.csv')
        assert len(records)==21 and len(set(map(key,records)))==21, f'incomplete worker: {worker}'
        for row in records:
            model=ROOT/row['model_path']
            assert sha256(model.read_bytes()).hexdigest()==row['model_sha256']
            if key(row) in rows:
                assert rows[key(row)]['model_sha256']==row['model_sha256'],key(row)
                continue
            assert key(row) not in {key(r) for r in pending}, 'overlapping worker seeds'
            pending.append(row)
    assert len(rows)+len(pending)==320,len(rows)+len(pending)
    # No archive file is changed until every worker and coefficient hash passes.
    for row in pending:
        model=ROOT/row['model_path']
        target=BASE/'models'/model.name
        if model!=target:
            shutil.copy2(model,target)
        row['model_path']=target.relative_to(ROOT).as_posix()
        rows[key(row)]=row
    assert len(rows)==320,len(rows)
    temporary=BASE/'rfm_raw.csv.tmp'
    with temporary.open('w',newline='') as f:
        records=[rows[k] for k in sorted(rows)]
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)
    temporary.replace(BASE/'rfm_raw.csv')
    for worker in workers:
        shutil.rmtree(worker)
    print('Merged 320 RFM observations; coefficient hashes preserved.')
if __name__=='__main__':
    main()
