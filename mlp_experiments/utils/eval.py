# %%
import copy
import numpy as np
import torch
import torch.nn.functional as F


def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    return (np.log(y[0])-np.log(y[-1]))/(np.log(x[0])-np.log(x[-1]))


def scaling_law(x,y):
    exp1 = exponent(x,y)
    const =  y[0] * x[0]**(-exp1)
    return (lambda inp: const * inp**(exp1))


def compute_feature_sing_vals(batches, model, out_layer=None,device=None,full_return = False, percentile_pcs = [0.5,0.9,0.95,0.99,0.999], num_batches = None,num_gpus = None):
    """
    Compute feature rank in all layers. Source: https://github.com/tml-epfl/sam-low-rank-features/blob/main/classification_tasks/train.py
    Def (feature rank): Given layer, min number of singular values to reach 99% of feature variance explained.
    """
    with torch.no_grad():
        phi_list = []
        for i, (X, y) in enumerate(batches):
            if (i < num_batches) or (num_batches is None):
                if device is not None:
                    if num_gpus is not None:
                        X, y = X.to(device), y.to(f'cuda:{num_gpus-1}')
                    else:
                        X, y = X.to(device), y.to(device)
                phi = model(X, out_layer = out_layer).detach().cpu().numpy()
                phi_list.append(phi)

        phi = np.concatenate(phi_list,axis=0)
        phi = phi.reshape(phi.shape[0], np.prod(phi.shape[1:]))
        phi = phi - np.mean(phi, axis=1, keepdims=True)
        try:
            feature_sing_vals = np.linalg.svd(phi, compute_uv=False)
        except:
            return [np.nan], np.nan, [np.nan]
        if full_return:
            return feature_sing_vals
        else:
            total=np.sum(feature_sing_vals**2)
            feat_ranks = [np.sum(np.cumsum(feature_sing_vals**2) <= total * percentile) + 1 for percentile in percentile_pcs]
            return feat_ranks, total, feature_sing_vals[:feat_ranks[-2]] # return singvals until 99% coverage
            


# get Hessian norm by: init random direction, power iterate until 0.001 relative error,
# approximate Hessian-vector product Hv with difference quotient: (Lossgrad(params+eps*v)-Lossgrad(params-eps*v))/(2*eps)
class Hessian_eval(object):
    """
    Compute the approximate Hessian spectral norm, gradient norm and alignment between gradient and leading Hessian eigenvector.
    """
    
    def __init__(self,model, loss_fn, data, target, max_iter=1000, diff_quot_stepsize=1e-3, eps_norm = 1e-3, to_cpu=False):
        super().__init__()
        with torch.no_grad():
            self.model = copy.deepcopy(model)
            self.frozen_statedict = copy.deepcopy(self.model.state_dict())
        self.loss_fn = loss_fn
        self.data = data
        self.target = target
        self.max_iter = max_iter
        self.diff_quot_stepsize = diff_quot_stepsize
        self.eps_norm = eps_norm
        self.to_cpu = to_cpu
        self.current_norm, self.grad_norm, self.grad_alignment, self.vec = None, None, None, None

    def eval_grad(self, state_dict=None):
        # compute param grad on (data, target)
        if state_dict is not None:
            model = copy.deepcopy(self.model)
            model.load_state_dict(state_dict)
        else:
            model=self.model
        model.train()
        model.zero_grad()
        output=model(self.data)
        loss=self.loss_fn(output,self.target)
        loss.backward()
        grads=[]
        for name, param in model.named_parameters():
            # for p in layer['linear'].parameters():
            grads.append(param.grad)
        model.eval()
        model.zero_grad()
        del model
        return grads # grads shape e.g. 3-layer MLP on CIFAR10: torch.Size([64, 3072]), [64, 64], [10, 64]

    def hessian_vector_prod(self, vec, num_eps=1e-8):
        # approximate Hessian-vector product H vec via difference quotient
        pert_statedict = copy.deepcopy(self.frozen_statedict)
        for i, key in enumerate(self.frozen_statedict):
            pert_statedict[key] = self.frozen_statedict[key] + self.diff_quot_stepsize * vec[i] # here plus
        plus_grads = self.eval_grad(pert_statedict)
        for i, key in enumerate(self.frozen_statedict):
            pert_statedict[key] = self.frozen_statedict[key] - self.diff_quot_stepsize * vec[i] # here minus
        minus_grads = self.eval_grad(pert_statedict)
        return [(plus_grads[i]-minus_grads[i])/(2*self.diff_quot_stepsize+num_eps) for i in range(len(plus_grads))]

    def iter(self, v):
        prods = self.hessian_vector_prod(v)
        norms = [torch.linalg.norm(prod) for prod in prods]
        total_norm = torch.sqrt(torch.sum(torch.tensor([norm**2 for norm in norms])))
        new_v = [prod/total_norm for prod in prods]
        return new_v, total_norm, norms

    def eval(self,prints=False):
        grads = self.eval_grad()
        
        # sample uniformly random direction in each layer
        vec = []
        for i, shape in enumerate([grad.shape for grad in grads]):
            rand_dir = torch.randn(np.prod(shape))
            rand_dir = F.normalize(rand_dir,dim=0)
            vec.append(rand_dir.reshape(shape)) # this is how Long does it, but not a proper normalization
        
        # perform power iteration
        t=0
        last_norm = None
        current_norm = 1.0
        while ((last_norm is None) or ((t<self.max_iter) and abs(current_norm-last_norm)>current_norm*self.eps_norm and abs(current_norm)>self.eps_norm)):
            vec, total_norm, norms = self.iter(vec)
            last_norm = current_norm
            current_norm = total_norm.detach().cpu().numpy()
            t += 1
            
        if prints:
            if t>=self.max_iter:
                print(f'Hessian eval:: Reached max_iter={self.max_iter} without converging up to {self.eps_norm}.')
            else:
                print(f'Hessian eval:: Converged to {abs(current_norm-last_norm)/current_norm} relative norm error after {t} iterations.')
        
        # compute alignment with gradient vector
        grad_norm = torch.sqrt(torch.sum(torch.tensor([torch.linalg.norm(grad)**2 for grad in grads])))
        grad_alignment = sum([torch.sum(vec[i]*grads[i]) for i in range(len(grads))])/grad_norm.item() # sum_layers scalar_product(grad_layer, Hessian_eigvec_layer)

        self.current_norm, self.grad_norm, self.grad_alignment = current_norm.item(), grad_norm.detach().cpu().numpy().item(), grad_alignment.detach().cpu().numpy().item()
        self.spectralnorms = [norm.detach().cpu().numpy().item() for norm in norms]
        if self.to_cpu:
            self.vec = vec.detach().cpu().numpy()
        else:
            self.vec = vec
        del self.model
        del self.frozen_statedict
        return self.current_norm, self.grad_norm, self.grad_alignment, self.vec, self.spectralnorms

# Hess_eval=Hessian_eval(model,loss_fn, data, target, max_iter=1000, diff_quot_stepsize=1e-3, eps_norm = 1e-3, to_cpu=False)
# spec_norm, grad_norm, grad_alignment, spec_vec, spec_norms = Hess_eval.eval()

def theoretical_SAM_EOS(lr,rho,gradnorm, ratio = False):
    """
    https://arxiv.org/pdf/2309.12488.pdf
    if ratio: SAM_EOS/SGD_EOS, else: SAM_EOS
    """

    if lr<=0:
        print(f'SAM EOS:: Need positive learning rate, not {lr}')
    if rho<=0:
        print(f'SAM EOS:: Need positive rho, not {rho}')
    if gradnorm<=0:
        print(f'SAM EOS:: Need positive grad norm, not {gradnorm}')
    try:
        if ratio:
            return lr*gradnorm/(4*rho) *(np.sqrt(1+8*rho/(lr*gradnorm)) - 1)
        else:
            return gradnorm/(2*rho) *(np.sqrt(1+8*rho/(lr*gradnorm)) - 1)
    except:
        return np.nan

# %%
