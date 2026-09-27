"""Build clean submission archives and verify them in an OS temporary directory.

Only local files are used. No network request, commit, tag, or publication is made.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'__pycache__', '.DS_Store', '.git'}
BUILD_SUFFIXES = {'.aux', '.log', '.fls', '.fdb_latexmk', '.synctex.gz', '.out', '.toc', '.blg', '.bbl', '.pyc'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_tree(source, target):
    for p in sorted(source.rglob('*')):
        relative = p.relative_to(source)
        if not p.is_file() or any(part in EXCLUDED for part in relative.parts):
            continue
        if p.suffix in BUILD_SUFFIXES or p.name.endswith('.synctex.gz'):
            continue
        out = target / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def inventory(root):
    excluded = {'manifest/package_inventory.csv', 'manifest/SHA256SUMS'}
    rows = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file()
            and str(p.relative_to(root)) not in excluded]
    (root / 'manifest').mkdir(exist_ok=True)
    with (root / 'manifest/package_inventory.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'bytes', 'sha256'])
        w.writeheader()
        w.writerows(rows)
    (root / 'manifest/SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in rows))


def archive(source, destination):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(source))


def build():
    stage = Path(tempfile.mkdtemp(prefix='cga_release_build_'))
    full = stage / 'full'
    full.mkdir()
    for name in ['code', 'config', 'data', 'result', 'tex', 'supplement']:
        copy_tree(ROOT / name, full / name)
    # Use the R5-rebuilt numerical artifacts rather than old cached figures.
    rebuilt = ROOT / 'tmp/R5_rebuilt_artifacts/tex'
    if rebuilt.is_dir():
        for name in ['generated', 'figures']:
            copy_tree(rebuilt / name, full / 'tex' / name)
    # Keep only the recursively included TeX sources; the standalone chapter is external.
    active = set()
    def visit(p):
        p = p.resolve()
        if p in active:
            return
        active.add(p)
        for name in re.findall(r'\\input\{([^}]+)\}', p.read_text()):
            q = full / 'tex' / name
            if not q.suffix:
                q = q.with_suffix('.tex')
            if q.is_file():
                visit(q)
    visit(full / 'tex/main.tex')
    for p in (full / 'tex/sections').glob('*.tex'):
        if p.resolve() not in active:
            p.unlink()  # Only a newly created staging directory.
    for name in ['main.tex', 'chapter5_theory.tex']:
        shutil.copy2(ROOT / name, full / name)
    shutil.copy2(ROOT / 'reports/package_README.md', full / 'README.md')
    (full / 'markdown').mkdir()
    for name in ['referee_response.md', 'review_resolution_matrix.csv']:
        shutil.copy2(ROOT / 'reports' / name, full / 'markdown' / name)
    (full / 'reports').mkdir()
    for name in ['R5_acceptance.json', 'R5_external_artifacts.json', 'R5_external_diagnostics.json']:
        shutil.copy2(ROOT / 'reports' / name, full / 'reports' / name)
    copy_tree(ROOT / 'reports/R5_external_logs', full / 'reports/R5_external_logs')
    (full / 'manifest').mkdir()
    for name in ['experiment_index_v2.csv', 'experiment_index.csv', 'figure_table_map.csv',
                 'restored_binary_inputs.csv', 'theorem_label_map.csv']:
        shutil.copy2(ROOT / 'manifest' / name, full / 'manifest' / name)
    artifact_manifest = ROOT / 'tmp/R5_rebuilt_artifacts/manifest/figure_table_manifest.csv'
    if artifact_manifest.is_file():
        shutil.copy2(artifact_manifest, full / 'manifest/figure_table_manifest.csv')
    inventory(full)
    overleaf = stage / 'overleaf'
    overleaf.mkdir()
    for name in ['tex', 'supplement']:
        copy_tree(full / name, overleaf / name)
    for name in ['main.tex', 'chapter5_theory.tex', 'README.md']:
        shutil.copy2(full / name, overleaf / name)
    (overleaf / 'config').mkdir()
    shutil.copy2(full / 'config/repository_link.json', overleaf / 'config/repository_link.json')
    inventory(overleaf)
    submit = ROOT / 'submit'
    if submit.exists():
        backup = ROOT / 'tmp' / ('submit_before_R7_R9_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        backup.parent.mkdir(exist_ok=True)
        submit.rename(backup)
    shutil.copytree(full, submit)
    archive(full, ROOT / 'submit_full.zip')
    archive(overleaf, ROOT / 'submit_overleaf.zip')
    return stage


def verify():
    external = Path(tempfile.mkdtemp(prefix='cga_R9_external_'))
    logs = ROOT / 'reports/R9_logs'
    logs.mkdir(exist_ok=True)
    checks = []
    packages = []
    def run(name, command, cwd):
        proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=600)
        (logs / (name + '.log')).write_text(proc.stdout)
        checks.append({'name': name, 'passed': proc.returncode == 0, 'exit_code': proc.returncode})
        if proc.returncode:
            raise RuntimeError(name + ': ' + proc.stdout[-1500:])
    for name in ['overleaf', 'full']:
        path = ROOT / f'submit_{name}.zip'
        target = external / name
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            z.extractall(target)
        rows = list(csv.DictReader((target / 'manifest/package_inventory.csv').open()))
        expected = {r['path'] for r in rows} | {'manifest/package_inventory.csv', 'manifest/SHA256SUMS'}
        actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
        assert actual == expected
        assert all(sha(target / r['path']) == r['sha256'] and
                   (target / r['path']).stat().st_size == int(r['bytes']) for r in rows)
        checks.append({'name': name + '_CRC_and_all_file_hashes', 'passed': True, 'files': len(actual)})
        run(name + '_sha256sums', ['shasum', '-a', '256', '-c', 'manifest/SHA256SUMS'], target)
        run(name + '_compile', ['latexmk', '-xelatex', '-interaction=nonstopmode', '-halt-on-error',
                               '-outdir=build', 'main.tex'], target)
        log = (target / 'build/main.log').read_text(errors='replace')
        bad = re.findall(r'[^\n]*(?:undefined|multiply defined|Overfull|Missing character|LaTeX Error)[^\n]*', log, re.I)
        checks.append({'name': name + '_tex_warnings', 'passed': not bad, 'warnings': bad})
        if bad:
            raise RuntimeError(str(bad))
        info = subprocess.check_output(['pdfinfo', str(target / 'build/main.pdf')], text=True)
        pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
        assert pages < 40
        packages.append({'name': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                         'pages': pages, 'files': len(actual)})
    full = external / 'full'
    for mode in ['artifacts', 'diagnostics']:
        run(mode, [sys.executable, 'code/reproduce.py', '--mode', mode, '--package-root', '.',
                   '--output-root', str(external / ('output_' + mode))], full)
        report = external / ('output_' + mode) / f'reports/reproduction_{mode}.json'
        shutil.copy2(report, ROOT / f'reports/R9_{mode}.json')
    run('negative_and_rebuild_tests', [sys.executable, 'code/check_R5_reproduction.py'], full)
    shutil.copy2(full / 'reports/R5_acceptance.json', ROOT / 'reports/R9_failure_tests.json')
    assert all(x['passed'] for x in json.loads((ROOT / 'reports/R9_failure_tests.json').read_text())['checks'])
    shutil.copy2(full / 'build/main.pdf', ROOT / 'revised_paper.pdf')
    report = {'status': 'PASS', 'scope': 'local archives; remote verification waived by user',
              'external_directory': str(external), 'checks': checks, 'packages': packages}
    (ROOT / 'reports/R9_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'manifest/final_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'sha256', 'bytes', 'pages', 'files'])
        w.writeheader()
        w.writerows(packages)
    (ROOT / 'manifest/SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in packages))
    shutil.copy2(ROOT / 'submit/manifest/package_inventory.csv', ROOT / 'manifest/package_inventory.csv')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    if '--verify-only' not in sys.argv:
        print('Staging:', build(), flush=True)
    verify()
