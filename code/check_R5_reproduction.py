"""Integration tests on a disposable package outside the source repository."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='cga_R5_acceptance_') as temporary:
        base = Path(temporary).resolve()
        package = base / 'package'
        package.mkdir()
        for directory in ('code', 'config', 'data', 'tex'):
            shutil.copytree(ROOT / directory, package / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '.DS_Store'))
        for directory in ('section85', 'revision_20260914'):
            shutil.copytree(ROOT / 'result' / directory, package / 'result' / directory)
        shutil.copy2(ROOT / 'chapter5_theory.tex', package / 'chapter5_theory.tex')
        # Remove generated products while retaining the two audited manuscript tables
        # that are static inputs rather than outputs of generate_artifacts.py.
        static_tables = {}
        for name in ('transfer_margin_example.tex', 'unified_audit_summary.tex'):
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
            command = [sys.executable, str(package / 'code/reproduce.py'), '--mode', mode,
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
        require_files = ['tex/generated/ch8_p4_tables.tex', 'tex/generated/revision_continuation_summary.tex',
                         'tex/figures/cga/cga_p4_representative.pdf']
        assert all((artifacts / p).is_file() for p in require_files)
        diagnostics, d = run('diagnostics_recomputed', 'diagnostics', True)
        assert d['rfm']['successes'] == 304 and d['rfm']['failures'] == 9
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
        relative = 'data/derived/window_diagnostics/window_certificate_validation.json'
        validation = json.loads((package / relative).read_text())
        validation['passed'] = False
        changed_file(relative, json.dumps(validation), 'false_validation_fails', 'diagnostics')
        validation['passed'] = 'False'
        changed_file(relative, json.dumps(validation), 'string_validation_fails', 'diagnostics')
        protocol = json.loads((package / 'config/revision_protocol.json').read_text())
        checkpoint = protocol['runs']['base_pure']['parent'] + '/checkpoint.npz'
        changed_file(checkpoint, None, 'missing_checkpoint_fails', 'diagnostics')
        run('missing_config_fails', 'artifacts', False, ('--config', 'config/absent.json'))
        # Present but invalid configuration must be parsed, not silently ignored.
        changed_file('config/reproduction.json', '{"schema_version":"invalid"}', 'invalid_config_fails', 'artifacts')
        for mode, source in [('artifacts', artifacts), ('diagnostics', diagnostics)]:
            for log in (source / 'reports/r5_logs').glob('*.log'):
                target = ROOT / 'reports/R5_external_logs' / log.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, target)
            target = ROOT / 'reports' / f'R5_external_{mode}.json'
            target.write_bytes((source / f'reports/reproduction_{mode}.json').read_bytes())
        payload = dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL', checks=checks,
                       artifacts=a['manuscript_dependencies'], rfm=d['rfm'],
                       code_sha256=digest(ROOT / 'code/reproduce.py'))
        (ROOT / 'reports/R5_acceptance.json').write_text(json.dumps(payload, indent=2) + '\n')
        print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
