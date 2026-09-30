#!/usr/bin/env python3
"""Verify that each N=32 continuation starts from the archived prefix."""
from pathlib import Path
import csv, hashlib, json, sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/cga_refactor/src'))
from cga_refactor.dictionary import sample_pool_parameters, canonicalize_pool

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out = {}
    protocol = json.loads((ROOT / 'config/continuation_protocol.json').read_text())
    for name in ("base_pure", "epsilon_01"):
        rev = ROOT / "data/raw/continuation_20260914" / name
        spec = protocol['runs'][name]
        parent = ROOT / spec['parent']
        rev_hist = (rev / "history.csv").read_bytes()
        par_hist = (parent / "history.csv").read_bytes()
        states_parent = {p.name for p in (parent / "states").glob("accepted_*.npz") if int(p.stem.split('_')[-1]) <= 32}
        states_rev = {p.name for p in (rev / "states").glob("accepted_*.npz") if int(p.stem.split('_')[-1]) <= 32}
        input_hashes_match = all(sha(ROOT / p) == h for p, h in spec['input_sha256'].items())
        checkpoint_checks = []
        for run, n in ((parent, 32), (rev, 64)):
            with (run / 'history.csv').open(newline='') as handle:
                history = list(csv.DictReader(handle))
            accepted = [int(r['accepted_pool_index']) for r in history if r['accepted'] == 'True']
            consumed = {int(r['nominal_pool_index']) for r in history if r['nominal_pool_index'] not in ('', 'None')}
            with np.load(run / 'checkpoint.npz') as ck, np.load(run / 'candidate_pool.npz') as pool, np.load(run / f'states/accepted_{n:04d}.npz') as state:
                checkpoint_checks.append(bool(int(ck['accepted']) == n and int(ck['attempted']) == int(ck['history_rows']) == len(history)
                    and np.array_equal(ck['accepted_indices'], accepted)
                    and set(np.flatnonzero(ck['candidate_consumed'])) == consumed
                    and np.array_equal(ck['coefficients'], state['coefficients'])
                    and all(np.array_equal(pool[k][accepted], state[k]) for k in ('w', 'b', 'scales'))))
        cfg = json.loads((parent / 'config.json').read_text())
        pool_checks = {}
        for kind in ('candidate', 'reference'):
            size = cfg['pool'][kind + '_size']
            seed = cfg['seed'] + (1009 if kind == 'reference' else 0)
            w, b = sample_pool_parameters(1, size, cfg['pool']['sampler'], seed, SimpleNamespace(**cfg['pool']))
            w, b, original = canonicalize_pool(w, b)
            with np.load(parent / (kind + '_pool.npz')) as pool:
                pool_checks[kind] = bool(np.array_equal(w, pool['w']) and np.array_equal(b, pool['b']) and np.array_equal(original, pool['original_indices']))
        with np.load(ROOT / spec['source_vector']) as source, np.load(rev / 'checkpoint.npz') as ck:
            contains_all = set(ck['accepted_indices']).issubset(set(source['candidate_indices']))
            source_info = {'candidate_count': len(source['candidate_indices']), 'reference_count': len(source['reference_indices']),
                           'coefficient_count': len(source['coefficients']), 'all_64_selected_directions_in_dictionary': contains_all,
                           'B_matches_protocol': bool(np.isclose(np.abs(source['coefficients']).sum(), spec['B'], rtol=1e-12, atol=1e-12))}
        out[name] = {"parent": str(parent.relative_to(ROOT)), "history_prefix_match": rev_hist.startswith(par_hist),
                     'frozen_input_hashes_match': input_hashes_match,
                     'protocol_hash_matches_run': sha(ROOT / 'config/continuation_protocol.json') == (rev / 'protocol_sha256.txt').read_text().strip(),
                     'parent_and_final_checkpoints_match_history': all(checkpoint_checks),
                     'pool_generation_exactly_reproduced': all(pool_checks.values()), 'source_dictionary': source_info,
                     "parent_history_sha256": sha(parent / "history.csv"), "continuation_history_sha256": sha(rev / "history.csv"),
                     "states_1_32_match": states_parent == states_rev and len(states_parent) == 32 and all(sha(parent / "states" / n) == sha(rev / "states" / n) for n in states_parent),
                     "fixed_pools_match": all(sha(parent / n) == sha(rev / n) for n in ("candidate_pool.npz", "reference_pool.npz")),
                     "parent_states": len(states_parent),
                     "continuation_states": len(list((rev / "states").glob("accepted_*.npz"))),
                     "target_accepted": json.loads((rev / "summary.json").read_text())["accepted_atom_count"]}
    passed = all(v['frozen_input_hashes_match'] and v['protocol_hash_matches_run'] and v['parent_and_final_checkpoints_match_history']
                 and v['pool_generation_exactly_reproduced'] and v['source_dictionary']['all_64_selected_directions_in_dictionary']
                 and v['source_dictionary']['B_matches_protocol'] and v["history_prefix_match"] and v["states_1_32_match"]
                 and v["fixed_pools_match"] and v["continuation_states"] == 64 and v["target_accepted"] == 64 for v in out.values())
    payload = {"status": "PASS" if passed else "FAIL", "models": out,
               "protocol": json.loads((ROOT / "config/continuation_protocol.json").read_text())}
    target = ROOT / "result/continuation_20260914/prefix_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(target)})); return 0 if passed else 1

if __name__ == "__main__": raise SystemExit(main())
