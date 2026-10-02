#!/usr/bin/env python3
"""Inventory the current two-folder public package, excluding generated inventories."""
from __future__ import annotations
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'code/manifest/package_inventory.csv'
SKIP_SUFFIXES={'.aux','.log','.out','.fls','.fdb_latexmk','.synctex','.pyc','.blg'}
SKIP_NAMES={'SHA256SUMS','package_inventory.csv','public_source_inventory.csv','.DS_Store'}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='verify current inventories and artifact hashes without writing')
    args=parser.parse_args()
    published = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
    ).decode().split('\0'))
    files=[]
    for directory in ('code','paper'):
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file() or path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
                continue
            rel=path.relative_to(ROOT)
            if rel.as_posix() not in published:
                continue
            if any(p in {'__pycache__','.pytest_cache','.mpl','tex'} for p in rel.parts):
                continue
            files.append(path)
    files+=list(ROOT.glob('*.md'))
    rows=[]
    for path in sorted(set(files)):
        rel=path.relative_to(ROOT).as_posix()
        category=('paper-source' if path.suffix in {'.tex','.bib','.cls','.bst'} else
                  'figure' if '/figures/' in rel else 'data' if '/data/' in rel else
                  'configuration' if '/config/' in rel else
                  'program' if path.suffix=='.py' else 'record')
        rows.append(dict(schema_version='cga-public-inventory-v2',category=category,
                         path=rel,bytes=path.stat().st_size,
                         sha256=sha256(path.read_bytes()).hexdigest()))
    sums=''.join(f"{r['sha256']}  {r['path']}\n" for r in rows)
    if args.check:
        problems=[]
        expected=[{k:str(v) for k,v in row.items()} for row in rows]
        recorded=list(csv.DictReader(OUTPUT.open(newline='',encoding='utf-8')))
        if recorded!=expected:
            old={r['path']:r for r in recorded}
            new={r['path']:r for r in expected}
            problems += [f'inventory mismatch: {p}' for p in sorted(old.keys()|new.keys()) if old.get(p)!=new.get(p)]
        if (ROOT/'code/manifest/SHA256SUMS').read_text()!=sums:
            problems.append('SHA256SUMS differs from the current package')
        artifacts=list(csv.DictReader((ROOT/'code/manifest/figure_table_manifest.csv').open(newline='',encoding='utf-8')))
        for artifact in artifacts:
            path=ROOT/'code'/artifact['artifact']
            if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=artifact['output_sha256']:
                problems.append(f"artifact mismatch: {artifact['artifact']}")
            for source in artifact['input_files_sha256'].split(';'):
                relative, recorded_hash=source.rsplit('#',1)
                source_path=ROOT/'code'/relative
                if not source_path.is_file() or sha256(source_path.read_bytes()).hexdigest()!=recorded_hash:
                    problems.append(f'input mismatch: {relative}')
            if path.suffix=='.pdf':
                sidecar=path.with_suffix('.manifest.json')
                if not sidecar.is_file() or json.loads(sidecar.read_text())!=artifact:
                    problems.append(f"sidecar mismatch: {artifact['artifact']}")
        print(json.dumps(dict(status='FAIL' if problems else 'PASS',files=len(rows),
                              artifacts=len(artifacts),problems=problems),indent=2))
        if problems:
            raise SystemExit(1)
        return
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (ROOT/'code/manifest/SHA256SUMS').write_text(sums,encoding='utf-8')
    print(f'Inventoried {len(rows)} files; all paths are repository-relative.')
if __name__=='__main__':
    main()
