"""
Implements SAM, elementwise and layerwise ASAM with optional rescaling of perturbation denominator terms and optional perturbation norm evaluation.
Adapted from https://github.com/davda54/sam/blob/main/sam.py
"""

import torch
import numpy as np

class SAM(torch.optim.Optimizer):
    """
    Implements SAM, elementwise and layerwise ASAM with optional rescaling of perturbation denominator terms and optional perturbation norm evaluation.
    Adapted from https://github.com/davda54/sam/blob/main/sam.py
    """
    
    def __init__(self,
                 params,
                 base_optimizer,
                 rho=0,
                 adaptive=False,
                 layerwise=False,
                 last_layer_norm=False, #careful, here all last-layer parameters have to be in param_groups[-1]
                 gradnorm_scaling = None,
                 **kwargs):
        
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive,layerwise=layerwise, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.last_layer_norm = last_layer_norm # use ||x^L|| as proxy for gradient norm
        if gradnorm_scaling is None:
            self.gradnorm_scaling = [1 for _ in range(len(self.param_groups))]
        else:
            self.gradnorm_scaling = gradnorm_scaling


    @torch.no_grad()
    def first_step(self, zero_grad=False, return_gradnorm = False, return_perturbnorm = False, return_spectralnorm=False):
        """
        Perturbation step with optional perturbation norm evaluations.
        """
        if not self.last_layer_norm:
            grad_norm = self._grad_norm()
        else:
            # take the last layer grad_norm as approx of total grad_norm
            shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
            grads = []
            group = self.param_groups[-1]
            for p in group["params"]:
                if p.grad is not None:
                    if group["adaptive"]:
                        t_w= torch.abs(p)
                    elif group["layerwise"]:
                        t_w = torch.norm(p)
                    else:
                        t_w = 1.0
                    grads.append((t_w * self.gradnorm_scaling[-1] * p.grad).norm(p=2).to(shared_device))
            grad_norm = torch.stack(grads).norm(p=2)

        if return_perturbnorm: perturbfrobnorms = []
        if return_spectralnorm: perturbnorms = []
        for _, group in enumerate(self.param_groups):
            if not group['perturb']:
                if return_perturbnorm: perturbfrobnorms.append(0)
                if return_spectralnorm: perturbnorms.append(0)
                continue # these params should not affect the perturbation
            scale = group["rho"] / (grad_norm + 1e-12)

            if return_perturbnorm: these_perturbfrobnorms = []
            if return_spectralnorm: these_perturbnorms = []
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if group["adaptive"]:
                    t_w_sq= torch.pow(p, 2)
                elif group["layerwise"]:
                    t_w_sq = torch.norm(p)**2
                else:
                    t_w_sq = 1.0
                e_w = t_w_sq * p.grad * scale.to(p)
                if return_perturbnorm:
                    these_perturbfrobnorms.append(torch.norm(e_w).detach().cpu().numpy())
                if return_spectralnorm:
                    if 'bn' in group['name'] and '.weight' in group['name']:
                        thisnorm = np.max(np.abs(e_w.detach().cpu().numpy()))
                    elif len(e_w.size())==2:
                        thisnorm = torch.linalg.matrix_norm(e_w, ord=2).item()
                    else:
                        thisnorm = torch.norm(e_w).detach().cpu().numpy()
                    these_perturbnorms.append(thisnorm)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
            if return_perturbnorm:
                perturbfrobnorms.append(np.linalg.norm(these_perturbfrobnorms))
            if return_spectralnorm:
                perturbnorms.append(np.linalg.norm(these_perturbnorms))


        if zero_grad: self.zero_grad()
        if return_gradnorm: return grad_norm.detach().cpu().numpy()
        if return_perturbnorm and return_spectralnorm: return perturbnorms, perturbfrobnorms
        elif return_perturbnorm: return perturbfrobnorms
        elif return_spectralnorm: return perturbnorms

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Update/descent step."""
        for _, group in enumerate(self.param_groups):
            if not group['perturb']: continue
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


    def _grad_norm(self):
        """Computes the grad norm wrt the perturbed weights."""
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        
        grads = []
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                if (p.grad is not None and group["perturb"]):
                    if group["adaptive"]:
                        t_w = torch.abs(p)
                    elif group["layerwise"]:
                        t_w = torch.norm(p)
                    else:
                        t_w = 1.0
                    grads.append((t_w * self.gradnorm_scaling[i] * p.grad).norm(p=2).to(shared_device))
        norm = torch.stack(grads).norm(p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)