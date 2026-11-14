"""Same-state public aggregate geometry; no claims about learned utility."""
import math
import torch

def attenuation(norm_u,norm_m):
    if not math.isfinite(norm_u) or not math.isfinite(norm_m) or norm_u<=0 or norm_m<=0:
        raise ValueError('calibration requires positive finite public norms')
    c=norm_u/norm_m
    if not 0<c<=1:raise ValueError('requested attenuation must be in (0,1]; no silent upscaling/clamping')
    return float(c)

def from_inner_products(u2,m2,dot):
    if not all(math.isfinite(float(v)) for v in (u2,m2,dot)) or u2<=0 or m2<=0:
        return dict(status='UNDEFINED_ZERO_OR_NONFINITE_AGGREGATE')
    un,mn=math.sqrt(u2),math.sqrt(m2);cos=float(dot/(un*mn))
    if abs(cos)>1+2e-6:raise ValueError('invalid Gram / Cauchy-Schwarz violation')
    cos=max(-1.,min(1.,cos));a_len=un/mn;a_ls=max(0.,dot/m2);a_capped=min(1.,a_ls)
    return dict(status='PASS',cosine_U_M=cos,angle_U_M_degrees=math.degrees(math.acos(cos)),norm_matching_scalar=a_len,
                norm_matched_relative_residual=math.sqrt(max(0.,2*(1-cos))),
                best_nonnegative_L2_scalar=a_ls,best_nonnegative_L2_relative_residual=math.sqrt(max(0.,1+a_ls*a_ls*m2/u2-2*a_ls*dot/u2)),
                best_attenuation_L2_scalar=a_capped,best_attenuation_L2_relative_residual=math.sqrt(max(0.,1+a_capped*a_capped*m2/u2-2*a_capped*dot/u2)),
                raw_relative_difference=math.sqrt(max(0.,1+m2/u2-2*dot/u2)),
                note='norm matching is generally not the least-squares-optimal scalar')

def aggregate_diagnostics(means,c=None):
    # Means are already sums of FP32 per-record values. Use FP64 for small Gram / angles.
    names=list(means);vectors=torch.stack([means[k].double() for k in names]);gram=vectors@vectors.T
    G=gram.detach().cpu();i,j=names.index('U'),names.index('M');r=from_inner_products(float(G[i,i]),float(G[j,j]),float(G[i,j]))
    if r['status']!='PASS':raise RuntimeError('undefined public U/M aggregate geometry')
    u,m=means['U'].double(),means['M'].double();a=r['norm_matching_scalar'];actual=float((u-a*m).norm()/u.norm())
    if abs(actual-r['norm_matched_relative_residual'])>1e-7:raise RuntimeError('equal-length residual identity mismatch')
    result=dict(vector_names=names,gram_fp64=G.tolist(),norms_fp64={k:math.sqrt(max(0.,float(G[t,t]))) for t,k in enumerate(names)},U_M=r,
                individual_vectors_persisted=False,scope='same checkpoint and ordered public examples, before Gaussian noise and optimizer')
    if c is not None:
        sm=means['M_scaled'].double();target=means['U'].double();result['fixed_scale']=dict(c=float(c),scaled_M_over_U=float(sm.norm()/target.norm()),c_times_M_over_U=float(c*m.norm()/target.norm()),
            scaled_M_vs_cM_relative_roundoff=float((sm-c*m).norm()/max(float((c*m).norm()),1e-30)),
            near_public_norm_match_within_10pct=abs(float(sm.norm()/target.norm())-1)<=.10,
            scope='fixed c from public initialization, never re-fitted during training')
    return result
