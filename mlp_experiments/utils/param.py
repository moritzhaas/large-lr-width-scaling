# %%
"""
Computes the exponents for all choices of learning rate and perturbation radius parameterizations for each SAM variant.
"""
# get the b_l, c_l, d_l, d automatically for the parametrization of your choice:
# bc-parametrizations: mup, ntp, sp, sp_stable
# d-param: mpp, global, gb, naive, first_layer
# variants: sam, elementwise, layerwise

import numpy as np

def get_bcd(L,param='mup',perturb='mpp', variant= 'sam', multiplier_mup = False):
    """
    Computes the exponents for all choices of learning rate and perturbation radius parameterizations for each SAM variant.
    L is the number of hidden layers.
    """
    if multiplier_mup and param == 'mup':
        return [1/2 for _ in range(L+1)], [0 for _ in range(L+1)], [0 for _ in range(L+1)], 0
    return bl_param(L,param),cl_param(L,param),dl_param(L,param,perturb,variant), d_param(L,perturb,variant)

def bl_param(L,param='mup'):
    """initialization variance will be n ** (-2 bl)"""
    if param == 'mup' or param == 'mup_largelr' or param == 'mup_largeinput':
        out = [1/2 for _ in range(L+1)]
        out[0] = 0
        out[-1] = 1
    elif param == 'ntp' or param == 'sp' or param == 'sp_stable' or param=='mup_spllit' or param=='mup_spllit_largelr' or param=='sp_largeinput':
        out = [1/2 for _ in range(L+1)]
        out[0] = 0
    elif param=='llm':
        out = [0 for _ in range(L+1)]
        out[-1] = 1/2
    else:
        raise ValueError(f'{param} not defined.')
    return out

def cl_param(L,param='mup'):
    """learning rate will be lr * n ** (-cl)"""
    if param == 'mup' or param == 'mup_largeinput':
        out = [0 for _ in range(L+1)]
        out[0] = -1
        out[-1] = 1
    elif param == 'mup_spllit':
        out = [1/2 for _ in range(L+1)]
        out[0] = -1/2
        out[-1] = 1
    elif param == 'mup_spllit_largelr':
        out = [1/2 for _ in range(L+1)]
        out[0] = -1/2
        out[-1] = 1/2
    elif param == 'mup_largelr':
        out = [1 for _ in range(L+1)]
        out[0] = 0
        out[-1] = 0
    elif param == 'ntp':
        out = [1 for _ in range(L+1)]
        out[0] = 0
    elif param == 'sp' or param=='llm' or param=='sp_largeinput':
        out = [0 for _ in range(L+1)]
    elif param == 'sp_stable':
        out = [1 for _ in range(L+1)]
    else:
        raise ValueError(f'{param} not defined.')     
    return out

def dl_param(L,param='mup',perturb='mpp', variant='sam'):
    """layerwise perturbation scaling exponent.
    perturb in [naive, global, gb, mpp, first_layer]"""
    if perturb == 'naive':
        return [0 for _ in range(L+1)]
    bl,cl = bl_param(L,param),cl_param(L,param)
    c_theta = np.minimum(bl[-1],cl[-1]).item()
    if variant == 'sam':
        if perturb == 'mpp':
            out = [3/2-c_theta for _ in range(L+1)]
            out[0] = 1/2-c_theta
            out[-1] = 3/2
        elif perturb == 'global':
            out = [0 for _ in range(L+1)] # equiv. to [1/2 for _ in range(L+1)]
        elif perturb == 'gb':
            out = [1-c_theta for _ in range(L+1)]
            out[-1] = 1/2
            out[0] = 1/2-c_theta
        elif perturb == 'first_layer':
            out = [2-c_theta+1/2 for _ in range(L+1)]
            out[0] = -c_theta+1/2
            out[-1] = 2+1/2
        else:
            raise ValueError(f'Perturbation {perturb} not defined.')
    elif variant == 'elementwise':
        if param == 'mup' and perturb == 'mpp':
            out = [0 for _ in range(L+1)]
        else: return ValueError(f'{param} {perturb} not defined for variant {variant}.')
    elif variant == 'layerwise':
        if param == 'mup' and perturb == 'mpp':
            out = [1 for _ in range(L+1)]
            out[0] = 0
            out[-1] = 0
    return out

def d_param(L, perturb='mpp',variant='sam'):
    """Global perturbation scaling exponent."""
    if perturb == 'naive': return 0
    if variant == 'sam':
        if perturb == 'mpp' or perturb == 'first_layer':
            return -1. / 2
        else:
            return 1. / 2
    elif variant == 'elementwise':
        if perturb == 'mpp':
            return 1/2
        else: return ValueError(f'Perturbation {perturb} not defined for variant {variant}.')
    elif variant == 'layerwise':
        if perturb == 'mpp':
            return 0
        else: return ValueError(f'Perturbation {perturb} not defined for variant {variant}.')
    else:
        return ValueError(f'Variant {variant} not defined.')


# %%