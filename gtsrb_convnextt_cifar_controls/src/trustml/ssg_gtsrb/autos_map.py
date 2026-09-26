"""Unit-bounded AUTO-S queries; h is already beta-weighted, one final release."""
import math
import torch


def query(values, role, gamma, q=None):
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    g = values['g']
    if role in {'CleanDP', 'DPOnly'}:
        return g/(g.norm(dim=-1, keepdim=True)+gamma)
    h = values['h']
    if role == 'Joint':
        s = g+h
        return s/(s.norm(dim=-1, keepdim=True)+gamma)
    if role == 'Componentwise':
        if q is None or not 0 < q < 1:
            raise ValueError('allocation must lie in (0,1)')
        return ((1-q)*g/(g.norm(dim=-1, keepdim=True)+(1-q)*gamma)
                + q*h/(h.norm(dim=-1, keepdim=True)+q*gamma))
    raise ValueError('unsupported AUTO-S role')
