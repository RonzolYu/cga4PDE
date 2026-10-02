"""Integration tests on a disposable package outside the source repository."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(compile_paper=False):
    checks = []
    with tempfile.TemporaryDirectory(prefix='cga_reproduction_') as temporary:
        base = Path(temporary).resolve()
        package = base / 'package'
        package.mkdir()
        for directory in ('scripts', 'config', 'data', 'tex'):
            shutil.copytree(ROOT / directory, package / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
        for directory in ('section85', 'continuation_20260914'):
            shutil.copytree(ROOT / 'result' / directory, package / 'result' / directory)
        shutil.copy2(ROOT / 'chapter5_theory.tex', package / 'chapter5_theory.tex')
        # Remove generated products while retaining the two audited manuscript tables
        # that are static inputs rather than outputs of generate_artifacts.py.
        static_tables = {}
        for name in ('transfer_margin_example.tex', 'baseline_terminal.tex'):
            source = package / 'tex/generated' / name
            static_tables[name] = source.read_bytes()
        for name in ('generated', 'figures'):
            shutil.rmtree(package / 'tex' / name)
        (package / 'tex/generated').mkdir(parents=True)
        for name, payload in static_tables.items():
            (package / 'tex/generated' / name).write_bytes(payload)
        before = {str(p.relative_to(package)): digest(p) for p in package.rglob('*') if p.is_file()}

        def run(label, mode, expected_success, extra=()):
            output = base / label
            command = [sys.executable, str(package / 'scripts/reproduce.py'), '--mode', mode,
                       '--package-root', str(package), '--output-root', str(output), *extra]
            result = subprocess.run(command, cwd=base, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=600)
            report = json.loads((output / f'reports/reproduction_{mode}.json').read_text())
            passed = ((result.returncode == 0) == expected_success and
                      report['status'] == ('PASS' if expected_success else 'FAIL'))
            checks.append(dict(name=label, passed=passed, exit_code=result.returncode,
                               status=report['status'], error=report.get('error')))
            if not passed:
                raise AssertionError((label, result.stdout))
            return output, report

        artifacts, a = run('missing_targets_rebuilt', 'artifacts', True)
        require_files = ['tex/generated/ch8_p4_tables.tex', 'tex/generated/continuation_summary.tex',
                         'tex/figures/cga/cga_p4_representative.pdf']
        assert all((artifacts / p).is_file() for p in require_files)
        import csv
        with (artifacts/'manifest/figure_table_manifest.csv').open(newline='') as f:
            manifested = list(csv.DictReader(f))
        assert all((artifacts/r['artifact']).is_file() and
                   digest(artifacts/r['artifact'])==r['output_sha256'] for r in manifested)
        checks.append(dict(name='all_manifested_outputs_exported', passed=True, count=len(manifested)))
        tables = list((artifacts / 'tex/generated').glob('*.tex'))
        revision_tokens = (r'\revtext', r'\begin{revision}', r'\end{revision}', r'\colorbox{yellow')
        assert all(not any(token in path.read_text() for token in revision_tokens) for path in tables)
        from PIL import Image
        previews = list((artifacts / 'tex/figures').rglob('*.png'))
        for path in previews:
            with Image.open(path) as image:
                assert image.convert('RGB').getpixel((0, 0)) == (255, 255, 255), path
        checks.append(dict(name='clean_tables_and_figure_backgrounds', passed=True,
                           tables=len(tables), figure_previews=len(previews)))
        if compile_paper:
            latexmk = shutil.which('latexmk')
            if latexmk is None:
                raise RuntimeError('--compile-paper requires latexmk on PATH')
            manuscript = base / 'rebuilt_manuscript'
            shutil.copytree(package / 'tex', manuscript,
                            ignore=shutil.ignore_patterns('*.aux', '*.bbl', '*.blg', '*.log',
                                                         '*.fls', '*.fdb_latexmk', '*.out'))
            for directory in ('generated', 'figures'):
                shutil.copytree(artifacts / 'tex' / directory, manuscript / directory,
                                dirs_exist_ok=True)
            build = subprocess.run([latexmk, '-pdf', '-interaction=nonstopmode',
                                    '-halt-on-error', '-cd', 'main.tex'], cwd=manuscript,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=180)
            log = ROOT / 'logs/sisc_cga_rebuilt.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('\n'.join(line.rstrip() for line in build.stdout.splitlines()).rstrip() + '\n')
            assert build.returncode == 0, build.stdout[-4000:]
            final_log = (manuscript / 'main.log').read_text()
            assert not any(token in final_log for token in
                           ('Undefined control sequence', 'undefined references',
                            'undefined citations', 'Overfull', 'multiply defined'))
            from pypdf import PdfReader
            pages = len(PdfReader(manuscript / 'main.pdf').pages)
            checks.append(dict(name='rebuilt_clean_manuscript_compiles', passed=True,
                               exit_code=build.returncode, pages=pages,
                               log='logs/sisc_cga_rebuilt.log'))
        diagnostics, d = run('diagnostics_recomputed', 'diagnostics', True)
        declared=json.loads((package/'data/derived/experiments/experiment_validation.json').read_text())['rfm']
        assert d['rfm']['rows']==declared['rows']
        assert d['rfm']['failures']==declared['failures']
        assert d['rfm']['successes']==declared['rows']-declared['failures']
        assert d['rfm']['archived_models']==466
        after = {str(p.relative_to(package)): digest(p) for p in package.rglob('*') if p.is_file()}
        checks.append(dict(name='output_root_honored_source_unchanged', passed=before == after))
        assert before == after

        def changed_file(relative, content, label, mode):
            p = package / relative
            saved = p.read_bytes()
            try:
                if content is None:
                    p.unlink()
                else:
                    p.write_text(content)
                run(label, mode, False)
            finally:
                p.write_bytes(saved)

        changed_file('data/derived/experiments/cga_metrics_long.csv', None, 'missing_input_fails', 'artifacts')
        import csv
        with (package/'data/derived/experiments/rfm_multiseed_raw.csv').open(newline='') as f:
            model_path=next(csv.DictReader(f))['model_path']
        changed_file(model_path,None,'missing_comparison_model_fails','artifacts')
        relative = 'data/derived/window_diagnostics/window_certificate_validation.json'
        validation = json.loads((package / relative).read_text())
        validation['passed'] = False
        changed_file(relative, json.dumps(validation), 'false_validation_fails', 'diagnostics')
        validation['passed'] = 'False'
        changed_file(relative, json.dumps(validation), 'string_validation_fails', 'diagnostics')
        protocol = json.loads((package / 'config/continuation_protocol.json').read_text())
        checkpoint = protocol['runs']['base_pure']['parent'] + '/checkpoint.npz'
        changed_file(checkpoint, None, 'missing_checkpoint_fails', 'diagnostics')
        run('missing_config_fails', 'artifacts', False, ('--config', 'config/absent.json'))
        # Present but invalid configuration must be parsed, not silently ignored.
        changed_file('config/reproduction.json', '{"schema_version":"invalid"}', 'invalid_config_fails', 'artifacts')
        for mode, source in [('artifacts', artifacts), ('diagnostics', diagnostics)]:
            for log in (source / 'reports/reproduction_logs').glob('*.log'):
                target = ROOT / 'reports/reproduction_logs' / log.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, target)
            target = ROOT / 'reports' / f'reproduction_{mode}.json'
            target.write_bytes((source / f'reports/reproduction_{mode}.json').read_bytes())
        payload = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL', checks=checks,
                       artifacts=a['manuscript_dependencies'], rfm=d['rfm'],
                       code_sha256=digest(ROOT / 'scripts/reproduce.py'))
        (ROOT / 'reports/reproduction_acceptance.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-paper', action='store_true',
                        help='also compile the clean manuscript with rebuilt products; requires latexmk')
    main(compile_paper=parser.parse_args().compile_paper)
