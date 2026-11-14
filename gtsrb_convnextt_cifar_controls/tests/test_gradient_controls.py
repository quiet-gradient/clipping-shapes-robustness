"""CPU checks for the added scientific controls and public-geometry identities."""
import pytest
import torch

from trustml.cifar_interventions.mechanism import maps
from trustml.gradient_controls.mechanism import contribution,exact_unit,release
from trustml.gradient_controls.query_geometry import aggregate_diagnostics,attenuation

@pytest.mark.parametrize('map_name',['hard','autos'])
def test_new_controls_are_bounded_and_preserve_nonzero_U_directions(map_name):
    generator=torch.Generator().manual_seed(19)
    g=torch.randn(17,11,generator=generator,dtype=torch.float64)*4
    h=torch.randn(17,11,generator=generator,dtype=torch.float64)*3
    options=dict(map_name=map_name,bound=1.,q=.25,gamma=.1)
    j,u,_=maps(g,h,**options)
    m=contribution(g,h,method='M',**options)
    n=contribution(g,h,method='N',**options)
    cm=contribution(g,h,method='M_scaled',public_scale=.78,**options)
    for value in (m,n,cm):
        assert (value.norm(dim=-1)<=1+1e-12).all()
        torch.testing.assert_close(exact_unit(value),exact_unit(u))
    torch.testing.assert_close(m.norm(dim=-1),j.norm(dim=-1))
    torch.testing.assert_close(n.norm(dim=-1),torch.ones(17,dtype=torch.float64))
    torch.testing.assert_close(cm,.78*m)

def test_M_N_coincide_only_for_saturated_nonzero_joint_contributions():
    g=torch.tensor([[3.,0.],[.2,0.]],dtype=torch.float64)
    h=torch.tensor([[0.,2.],[0.,.1]],dtype=torch.float64)
    o=dict(map_name='hard',bound=1.,q=.5,gamma=.1)
    m=contribution(g,h,method='M',**o);n=contribution(g,h,method='N',**o)
    torch.testing.assert_close(m[0],n[0]);assert m[1].norm()<n[1].norm()

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_zero_and_tiny_direction_are_not_silently_clamped(dtype):
    tiny=torch.finfo(dtype).tiny
    v=torch.tensor([[0.,0.],[tiny,tiny]],dtype=dtype)
    u=exact_unit(v)
    assert torch.equal(u[0],torch.zeros(2,dtype=dtype))
    torch.testing.assert_close(u[1].norm(),torch.tensor(1.,dtype=dtype))

def test_zero_Componentwise_invalidates_unconditional_equal_magnitude_claim():
    # Equal component budgets cancel saturated opposing components even though
    # their original sum, and hence Joint, is nonzero.
    g=torch.tensor([[3.,0.]],dtype=torch.float64);h=torch.tensor([[-2.,0.]],dtype=torch.float64)
    o=dict(map_name='hard',bound=1.,q=.5,gamma=.1)
    j,u,_=maps(g,h,**o);m=contribution(g,h,method='M',**o)
    assert j.norm()>0 and u.norm()==0 and m.norm()==0

def test_attenuation_retains_the_original_absolute_noise():
    z=torch.tensor([2.,-3.],dtype=torch.float64);c=.78;sigma=1.123046875
    a=release(z,sigma=sigma,bound=1,denominator=512,generator=torch.Generator().manual_seed(4))
    b=release(c*z,sigma=sigma,bound=1,denominator=512,generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(a-z/512,b-c*z/512,rtol=1e-12,atol=1e-16)
    assert not torch.allclose(b,c*a)

def test_nonprivate_control_removes_both_bounding_and_noise():
    g=torch.tensor([[30.,4.]],dtype=torch.float64);h=torch.tensor([[10.,-4.]],dtype=torch.float64)
    v=contribution(g,h,method='nonprivate',map_name='hard',bound=1,q=.5,gamma=.1)
    torch.testing.assert_close(v,g+h);assert v.norm()>1
    torch.testing.assert_close(release(v.sum(0),sigma=0,bound=1,denominator=512,generator=None,nonprivate=True),torch.tensor([40.,0.],dtype=torch.float64)/512)

def test_public_aggregate_geometry_detects_angle_after_matching_length():
    means={'U':torch.tensor([1.,0.],dtype=torch.float64),'M':torch.tensor([1.,1.],dtype=torch.float64)}
    c=attenuation(float(means['U'].norm()),float(means['M'].norm()))
    means['M_scaled']=c*means['M'];d=aggregate_diagnostics(means,c)
    assert abs(d['fixed_scale']['scaled_M_over_U']-1)<1e-12
    assert abs(d['U_M']['angle_U_M_degrees']-45)<1e-10
    assert d['U_M']['norm_matched_relative_residual']>0

@pytest.mark.parametrize('scale',[0,-1,1.1,float('nan')])
def test_invalid_attenuation_is_rejected(scale):
    with pytest.raises(ValueError):
        contribution(torch.ones(1,2),torch.ones(1,2),method='M_scaled',public_scale=scale,map_name='autos',bound=1,q=.25,gamma=.1)
