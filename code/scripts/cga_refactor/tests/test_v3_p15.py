from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from cga_refactor.problems import (make_problem, exact_solution, source_term, energy,
    first_variation, coefficient_hvp, radial_map)
from cga_refactor.metrics import bregman_divergence, vp_map
from cga_refactor.quadrature import tensor_gauss_2d
from cga_refactor.step1 import feature_columns, _pool_scores
from cga_refactor.dictionary import build_pool, calibrate_pool, evaluate_atoms
from cga_refactor.config import config_for

def test_feature_layout_and_gram_2d():
    rng=np.random.default_rng(13)
    v=rng.normal(size=(11,4)); g=rng.normal(size=(11,4,2))
    rule=SimpleNamespace(weights=rng.uniform(.1,1,11))
    f=feature_columns(v,g,rule)
    exact=np.einsum('qi,qj,q->ij',v,v,rule.weights)+np.einsum('qid,qjd,q->ij',g,g,rule.weights)
    np.testing.assert_allclose(f.T@f,exact,rtol=1e-14)
    np.testing.assert_allclose(f[11:].reshape(11,2,4)/np.sqrt(rule.weights)[:,None,None],g.transpose(0,2,1))

@pytest.mark.parametrize('dim',[1,2])
def test_weak_load_derivative_and_zero(dim):
    rule=tensor_gauss_2d(3,3)
    if dim==1:
        rule=SimpleNamespace(points=rule.points[:,:1],weights=rule.weights)
    p=make_problem('pure_p',dim,1.5)
    ex=exact_solution(p,rule.points); load=source_term(p,rule.points)
    with np.errstate(all='raise'):
        assert np.all(radial_map(np.zeros((5,dim)),.5)==0)
        assert np.all(vp_map(np.zeros((5,dim)),1.5)==0)
    u,g=ex[0]*.7,ex[1]*.7
    v,h=ex[0]*.3,ex[1]*.3
    derivative=first_variation(p,u,g,v,h,load,rule)
    eps=1e-5
    fd=(energy(p,u+eps*v,g+eps*h,load,rule)-energy(p,u-eps*v,g-eps*h,load,rule))/(2*eps)
    np.testing.assert_allclose(derivative,fd,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(energy(p,u,g,load,rule)-energy(p,*ex[:2],load,rule),
                               bregman_divergence(p,u,g,ex,rule),atol=3e-15)
    with pytest.raises(ValueError):
        coefficient_hvp(p,None,None,None,None,None,None)

def test_fast_scores_equal_weak_directional_derivatives():
    p=make_problem('pure_p',2,1.5); rule=tensor_gauss_2d(4,3)
    cfg=replace(config_for('pure_p',2,profile='smoke'),p=1.5,relu_power=3)
    pool=build_pool(p,cfg,None,'candidate'); calibrate_pool(p,pool,rule)
    u,g,_=exact_solution(p,rule.points); state=SimpleNamespace(u=.4*u,grad_u=.4*g)
    load=source_term(p,rule.points)
    scores=_pool_scores(p,state,pool,rule,load,8)
    idx=np.flatnonzero(pool.available)
    v,h=evaluate_atoms(rule.points,pool.w[idx],pool.b[idx],pool.k)
    v-=pool.centers[idx]
    exact=np.abs(first_variation(p,state.u,state.grad_u,v,h,load,rule)/pool.scales[idx])
    np.testing.assert_allclose(scores[idx],exact,rtol=2e-13,atol=1e-14)
