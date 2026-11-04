"""Optional official PG learning rates shared by direct and configured API."""
import math


def recipe_overrides(*, lr=None, d_lr_mult=None, prior_lr_mult=None):
    if prior_lr_mult is not None:
        raise ValueError('adapter-only Cascade has no particle bank; prior_lr_mult is unavailable')
    result = {}
    for name,value in dict(lr=lr,d_lr_mult=d_lr_mult).items():
        if value is not None:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
            result[name]=float(value)
    return result
