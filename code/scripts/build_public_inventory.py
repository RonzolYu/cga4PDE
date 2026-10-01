#!/usr/bin/env python3
"""Inventory the current two-folder public package, excluding generated inventories."""
from __future__ import annotations
import csv
from hashlib import sha256
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'code/manifest/package_inventory.csv'
SKIP_SUFFIXES={'.aux','.log','.out','.fls','.fdb_latexmk','.synctex','.pyc','.blg'}
SKIP_NAMES={'SHA256SUMS','package_inventory.csv','public_source_inventory.csv','.DS_Store'}
def main():
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
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (ROOT/'code/manifest/SHA256SUMS').write_text(
        ''.join(f"{r['sha256']}  {r['path']}\n" for r in rows),encoding='utf-8')
    print(f'Inventoried {len(rows)} files; all paths are repository-relative.')
if __name__=='__main__':
    main()
