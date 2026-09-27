"""Read-only re-evaluation of saved p=4 states and finite-trajectory estimates.

Results are ordinary quadrature computations, not interval enclosures.
All source vectors are constructed at the entry, independently of subsequent errors.
"""
from pathlib import Path
import argparse
import csv
import json
from hashlib import sha256
import sys
import numpy as np
from scipy.linalg import qr

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK
sys.path.insert(0,str(WORK/'code/cga_refactor/src'))
from cga_refactor.dictionary import evaluate_atoms, load_pool, breakpoints_1d
from cga_refactor.quadrature import segmented_gauss_1d
from cga_refactor.solver import evaluate_saved_model
from cga_refactor.problems import make_problem, exact_solution, source_term, energy
from cga_refactor.metrics import compute_metrics

OUT=WORK/'data/derived/section85'
ARCHIVE=WORK/'data/raw/sensitivity'
MODEL_LABELS={
    'base_pure':'Pure p=4',
    'epsilon_01':'Regularized p=4, epsilon=0.1',
    'epsilon_001':'Regularized p=4, epsilon=0.01',
    'epsilon_1':'Regularized p=4, epsilon=1',
    'pool_256':'Pure p=4, candidate pool 256',
    'pool_1024':'Pure p=4, candidate pool 1024',
    'quadrature_4':'Pure p=4, training order 4',
    'quadrature_10':'Pure p=4, training order 10',
    'tolerance_loose':'Pure p=4, projection tolerance 1e-4',
    'tolerance_tight':'Pure p=4, projection tolerance 1e-8',
}

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run_path(name):
    paths=list((ARCHIVE/name).glob('*/summary.json'))
    if len(paths)!=1:raise RuntimeError((name,paths))
    return paths[0].parent

def gradient_matrix(pool, points):
    _,g=evaluate_atoms(points,pool.w,pool.b,pool.k)
    return g[:,:,0]

def finite_model(name, order):
    run=run_path(name)
    cfg=json.loads((run/'config.json').read_text())
    problem=make_problem(cfg['model'],1,4,cfg['epsilon'])
    candidate,reference=load_pool(run/'candidate_pool.npz'),load_pool(run/'reference_pool.npz')
    usable=np.flatnonzero(np.isfinite(candidate.scales)&(candidate.scales>1e-12))
    refusable=np.flatnonzero(np.isfinite(reference.scales)&(reference.scales>1e-12))[:128]
    points=np.unique(np.r_[breakpoints_1d(candidate),breakpoints_1d(reference),.5])
    rule=segmented_gauss_1d(points,order)
    b=exact_solution(problem,rule.points)[1][:,0]
    weight=3*b*b+(float(cfg['epsilon'])**2 if cfg['epsilon'] is not None else 0)
    root_weight=np.sqrt(rule.weights*weight)
    dc=gradient_matrix(candidate,rule.points)[:,usable]
    dr=gradient_matrix(reference,rule.points)[:,refusable]
    dictionary=np.column_stack([dc,dr])
    norms=np.linalg.norm(root_weight[:,None]*dictionary,axis=0)
    if np.any(norms<=0):raise RuntimeError('Null frozen dictionary element')
    dictionary/=norms
    F=root_weight[:,None]*dictionary
    target=root_weight*b
    models=sorted((run/'states').glob('accepted_*.npz'))
    saved=[]
    for model in models:
        m=int(model.stem.split('_')[-1])
        if not 8<=m<=32:continue
        with np.load(model) as z:
            _,g=evaluate_atoms(rule.points,z['w'],z['b'],int(z['k']))
            active=g[:,:,0]/z['scales']
            coefficients=z['coefficients'].copy()
        A=root_weight[:,None]*active
        Q,R=qr(A,mode='economic')
        singular=np.linalg.svd(R,compute_uv=False)
        if singular[-1] <= 1e-13*singular[0]:
            raise RuntimeError(f'Full active-space projection unresolved at m={m}; no rank truncation allowed')
        projected=Q@(Q.T@target)
        residual=target-projected
        actual=A@coefficients
        D2=float(np.sum((actual-projected)**2))
        h=active@coefficients-b
        remainder=float(rule.weights@(b*h**3+.25*h**4))
        delta=float(rule.weights@(.25*h*h*((h+2*b)**2+2*b*b)))
        if cfg['epsilon'] is not None:delta+=float(.5*cfg['epsilon']**2*(rule.weights@(h*h)))
        saved.append({'m':m,'Q':Q,'R':R,'r2':float(residual@residual),'D2':D2,
                      'residual':residual,'remainder':remainder,'delta':delta,
                      'condition':float(singular[0]/singular[-1]),'active':active})
    entry=saved[0]
    F0=F-entry['Q']@(entry['Q'].T@F)
    H=F0.T@F0; rhs=F0.T@entry['residual']
    # Fixed entry-only lasso penalties; choose the smallest listed penalty, not a tail fit.
    sources=[]
    a=np.zeros(H.shape[0])
    for penalty in [1e-2,1e-4,1e-6]:
        grad=H@a-rhs
        for sweep in range(2500):
            largest=0.0
            for i in range(a.size):
                if H[i,i]<=1e-18:continue
                value=a[i]-grad[i]/H[i,i]
                new=np.sign(value)*max(abs(value)-penalty/H[i,i],0)
                change=new-a[i]
                if change:
                    a[i]=new;grad+=change*H[:,i];largest=max(largest,abs(change))
            if largest<1e-10:break
        sigma=float(np.linalg.norm(entry['residual']-F0@a))
        sources.append({'penalty':penalty,'B':float(np.abs(a).sum()),'sigma':sigma,
                        'sweeps':sweep+1,'coefficients':a.copy()})
    chosen=sources[-1]
    np.savez(OUT/f'{name}_source_q{order}.npz',coefficients=chosen['coefficients'],
             norms=norms,candidate_indices=usable,reference_indices=refusable)
    (OUT/f'{name}_sources_q{order}.json').write_text(json.dumps(
        [{k:v for k,v in z.items() if k!='coefficients'} for z in sources],indent=2)+'\n')
    rows=[]; W=0.0; previous=None
    entry_r2=float(saved[0]['r2'])
    sigma2=float(chosen['sigma'])**2
    initial_bound = (sigma2 + 1.0/(1.0/(entry_r2-sigma2)+W/(4*chosen['B']**2))
                     if entry_r2>sigma2 else entry_r2)
    q=max(np.sqrt(z['D2']/z['r2']) for z in saved if z['r2']>0)
    rho=max(abs(z['remainder'])/(z['r2']+z['D2']) for z in saved)
    for index,z in enumerate(saved):
        U=(sigma2+4*chosen['B']**2/W) if W>0 else z['r2']
        U_init=(sigma2+1.0/(1.0/(entry_r2-sigma2)+W/(4*chosen['B']**2))
                if entry_r2>sigma2 else entry_r2) if W>0 else entry_r2
        U_monotone=min(entry_r2,U)
        U_used=min(entry_r2,U_init)
        energy_bound=(.5+rho)*(1+q*q)*U
        energy_bound_init=(.5+rho)*(1+q*q)*U_init
        energy_bound_monotone=(.5+rho)*(1+q*q)*U_monotone
        energy_bound_used=(.5+rho)*(1+q*q)*U_used
        row={'model':name,'N':z['m'],'order':order,'r2':z['r2'],'D2':z['D2'],
             'remainder':z['remainder'],'energy_gap':z['delta'],'B':chosen['B'],
             'sigma':chosen['sigma'],'sigma2':sigma2,'entry_r2':entry_r2,'W':W,
             'projection_upper_estimate':U,'projection_upper_init_estimate':U_init,
             'projection_upper_monotone_estimate':U_monotone,
             'projection_upper_used_estimate':U_used,
             'q':q,'rho':rho,'energy_upper_estimate':(.5+rho)*(1+q*q)*U,
             'energy_upper_init_estimate':energy_bound_init,
             'energy_upper_monotone_estimate':energy_bound_monotone,
             'energy_upper_used_estimate':energy_bound_used,
             'energy_additive_upper_estimate':.5*U+.5*z['D2']+abs(z['remainder']),
             'energy_additive_upper_init_estimate':.5*U_init+.5*z['D2']+abs(z['remainder']),
             'energy_bound_ratio':energy_bound/max(z['delta'],np.finfo(float).tiny),
             'energy_bound_init_ratio':energy_bound_init/max(z['delta'],np.finfo(float).tiny),
             'energy_bound_used_ratio':energy_bound_used/max(z['delta'],np.finfo(float).tiny),
             'decomposition_defect':z['delta']-(.5*(z['r2']+z['D2'])+z['remainder']),
             'active_condition':z['condition'],'theta':None,'ell2':None,
             'projection_drop_defect':None,'validation_level':'floating_quadrature_not_interval',
             'model_label':MODEL_LABELS.get(name,name)}
        if index+1<len(saved):
            nxt=saved[index+1]
            d=root_weight*nxt['active'][:,-1];d/=np.linalg.norm(d)
            score=abs(float(z['residual']@d)); S=float(np.max(np.abs(F.T@z['residual'])))
            innovation=d-z['Q']@(z['Q'].T@d);ell2=float(innovation@innovation)
            theta=score/S if S>0 else None
            row.update(theta=theta,ell2=ell2,
                       projection_drop_defect=z['r2']-nxt['r2']-score*score/ell2)
            if S>0 and ell2>0:W+=theta*theta/ell2
        rows.append(row)
    write_csv(OUT/f'{name}_finite_q{order}.csv',rows)
    initial_bound=rows[0]['projection_upper_init_estimate']
    print(json.dumps({'model':name,'order':order,'B':chosen['B'],'sigma':chosen['sigma'],
                      'q':q,'rho':rho,'W':W,'entry_projection_bound':initial_bound,
                      'last_error':rows[-1]['energy_gap'],
                      'last_bound':rows[-1]['energy_upper_estimate'],
                      'last_init_bound':rows[-1]['energy_upper_init_estimate'],
                      'max_decomposition_defect':max(abs(z['decomposition_defect']) for z in rows)}),flush=True)

def _frozen_system(run,order):
    """Rebuild the entry frozen system for one quadrature rule."""
    cfg=json.loads((run/'config.json').read_text())
    problem=make_problem(cfg['model'],1,4,cfg['epsilon'])
    candidate,reference=load_pool(run/'candidate_pool.npz'),load_pool(run/'reference_pool.npz')
    usable=np.flatnonzero(np.isfinite(candidate.scales)&(candidate.scales>1e-12))
    refusable=np.flatnonzero(np.isfinite(reference.scales)&(reference.scales>1e-12))[:128]
    points=np.unique(np.r_[breakpoints_1d(candidate),breakpoints_1d(reference),.5])
    rule=segmented_gauss_1d(points,order)
    b=exact_solution(problem,rule.points)[1][:,0]
    weight=3*b*b+(float(cfg['epsilon'])**2 if cfg['epsilon'] is not None else 0)
    root_weight=np.sqrt(rule.weights*weight)
    dc=gradient_matrix(candidate,rule.points)[:,usable]
    dr=gradient_matrix(reference,rule.points)[:,refusable]
    dictionary=np.column_stack([dc,dr])
    norms=np.linalg.norm(root_weight[:,None]*dictionary,axis=0)
    dictionary/=norms
    F=root_weight[:,None]*dictionary
    target=root_weight*b
    entry=run/'states'/'accepted_0008.npz'
    with np.load(entry) as z:
        _,g=evaluate_atoms(rule.points,z['w'],z['b'],int(z['k']))
        active=g[:,:,0]/z['scales']
    A=root_weight[:,None]*active
    Q,_=qr(A,mode='economic')
    F0=F-Q@(Q.T@F)
    residual=target-Q@(Q.T@target)
    return F0,residual

def crosscheck_sources():
    """Evaluate each entry-only coefficient vector on the other quadrature rule."""
    rows=[]
    for name in ['base_pure','epsilon_01']:
        run=run_path(name)
        systems={order:_frozen_system(run,order) for order in [16,20]}
        vectors={}
        for order in [16,20]:
            with np.load(OUT/f'{name}_source_q{order}.npz') as z:
                vectors[order]=z['coefficients']
        for source_order in [16,20]:
            a=vectors[source_order]
            for evaluation_order in [16,20]:
                F0,residual=systems[evaluation_order]
                sigma=float(np.linalg.norm(residual-F0@a))
                rows.append({
                    'model':name,
                    'model_label':MODEL_LABELS.get(name,name),
                    'source_order':source_order,
                    'evaluation_order':evaluation_order,
                    'B':float(np.abs(a).sum()),
                    'sigma':sigma,
                    'sigma2':sigma*sigma,
                    'coefficient_l2':float(np.linalg.norm(a)),
                    'coefficient_l1':float(np.abs(a).sum()),
                })
    write_csv(OUT/'source_quadrature_crosscheck.csv',rows)
    print('Wrote source quadrature cross-check',len(rows),'rows',flush=True)

def summarize_finite():
    """Write one compact table used by the paper and reports."""
    rows=[]
    for name in ['base_pure','epsilon_01']:
        for order in [16,20]:
            data=list(csv.DictReader((OUT/f'{name}_finite_q{order}.csv').open()))
            last=data[-1]
            rows.append({
                'model':name,
                'model_label':MODEL_LABELS.get(name,name),
                'order':order,
                'entry_N':int(data[0]['N']),
                'terminal_N':int(last['N']),
                'B':last['B'],
                'sigma':last['sigma'],
                'q':last['q'],
                'rho':last['rho'],
                'W':last['W'],
                'r_entry':data[0]['r2'],
                'projection_bound_C3':last['projection_upper_estimate'],
                'projection_bound_init':last['projection_upper_init_estimate'],
                'projection_bound_used':last['projection_upper_used_estimate'],
                'energy_bound_C3':last['energy_upper_estimate'],
                'energy_bound_init':last['energy_upper_init_estimate'],
                'energy_bound_used':last['energy_upper_used_estimate'],
                'energy_gap_terminal':last['energy_gap'],
                'energy_bound_over_gap':last['energy_bound_ratio'],
                'energy_bound_init_over_gap':last['energy_bound_init_ratio'],
                'energy_bound_used_over_gap':last['energy_bound_used_ratio'],
                'max_decomposition_defect':max(abs(float(r['decomposition_defect'])) for r in data),
                'max_projection_drop_defect':max(
                    abs(float(r['projection_drop_defect']))
                    for r in data if r['projection_drop_defect'] not in ('','None')),
                'validation_level':'floating_quadrature_not_interval',
            })
    write_csv(OUT/'finite_trajectory_summary.csv',rows)
    print('Wrote finite-trajectory summary',len(rows),'rows',flush=True)

def reevaluate():
    base=ARCHIVE
    runs=[p.parent for p in sorted(base.glob('*/*/summary.json'))]
    bp=[np.array([0,.5,1])]
    for run in runs:
        for name in ['candidate_pool.npz','reference_pool.npz']:
            bp.append(breakpoints_1d(load_pool(run/name)))
    points=np.unique(np.concatenate(bp))
    rules=[segmented_gauss_1d(points,o) for o in [16,20]]
    rows=[]
    for run in runs:
        cfg=json.loads((run/'config.json').read_text())
        problem=make_problem(cfg['model'],1,cfg['p'],cfg['epsilon'])
        for rule,order in zip(rules,[16,20]):
            ex=exact_solution(problem,rule.points);f=source_term(problem,rule.points)
            ee=energy(problem,*ex[:2],f,rule)
            for model in sorted((run/'states').glob('accepted_*.npz')):
                u,g=evaluate_saved_model(model,problem,rule.points)
                value=energy(problem,u,g,f,rule)
                metrics=compute_metrics(problem,u,g,ex,rule,value,ee)
                h=g[:,0]-ex[1][:,0];b=ex[1][:,0]
                stable=float(rule.weights@(.25*h*h*((h+2*b)**2+2*b*b)))
                if cfg['epsilon'] is not None:stable+=float(.5*cfg['epsilon']**2*(rule.weights@(h*h)))
                rows.append({'setting':run.parent.name,'N':int(model.stem.split('_')[-1]),
                             'order':order,'points':len(rule.points),'source':str(model.relative_to(ROOT)),
                             'sha256':sha256(model.read_bytes()).hexdigest(),
                             'stable_bregman':stable,**metrics})
    write_csv(OUT/'common_quadrature_states.csv',rows)
    terminal=[z for z in rows if z['N']==32]
    write_csv(OUT/'common_quadrature_terminal.csv',terminal)
    print('Re-evaluated',len(rows),'state-rule pairs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['replay','finite','crosscheck','summary','all'],default='all')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.part in ['replay','all']:reevaluate()
    if args.part in ['finite','all']:
        for name in ['base_pure','epsilon_01']:
            for order in [16,20]:finite_model(name,order)
    if args.part in ['crosscheck','all']:crosscheck_sources()
    if args.part in ['summary','all']:summarize_finite()
