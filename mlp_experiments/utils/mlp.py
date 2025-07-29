import fnmatch
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from utils import get_bcd
    
    
class MLP(torch.nn.Module):
  """
  Implements MLPs specifying LR and RHO parameterization. Initializes the MLP according to parameterization. Allows weight multipliers.
  """
  
  def __init__(self,
               in_size: int,
               out_size: int,
               hidden_size: int,
               n_hidden_layers: int,
               use_bias: bool = False,
               activation = F.relu,
               out_activation = None,
               res_connections = False,
               parameterization: str = 'mup',
               perturbation: str = 'mpp',
               init_var = 2,
               flat_indim: int = None,
               multipl_inout=None,# list of multipliers, so far only idx 0 and -1 supported
               large_bias_init = False,
               ll_zero_init = False,
               initvarnorm = False,
               multiplier_mup = False,
               p_dropout = None):
    super().__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.hidden_size = hidden_size
    self.n_hidden_layers = n_hidden_layers
    self.n_layers = n_hidden_layers + 1
    self.use_bias = use_bias
    self.activation = activation
    self.out_activation = out_activation
    self.res_connections = res_connections
    self.parameterization = parameterization
    self.perturbation = perturbation
    self.init_var = init_var
    self.flat_indim = flat_indim
    self.multipl_inout = multipl_inout
    self.ll_zero_init = ll_zero_init
    self.large_bias_init = large_bias_init
    self.initvarnorm = initvarnorm
    self.multiplier_mup = multiplier_mup
    self.p_dropout = p_dropout
    
    in_sizes = [self.in_size] + [self.hidden_size] * self.n_hidden_layers
    out_sizes = [self.hidden_size] * self.n_hidden_layers + [self.out_size]
    
    self.layers = []
    for i in range(self.n_hidden_layers+1):
      layer = {}
      layer['linear'] = torch.nn.Linear(in_sizes[i], out_sizes[i], bias=self.use_bias)
      self.add_module(f"linear_{i+1}", layer['linear'])
      if self.p_dropout is not None and i < self.n_hidden_layers:
        layer['dropout'] = torch.nn.Dropout(p=self.p_dropout)
        self.add_module(f"dropout_{i+1}", layer['dropout'])
      layer['activation'] = self.activation if i < self.n_hidden_layers  else self.out_activation
      self.layers.append(layer)
      
    bl, cl, dl, d = get_bcd(L=self.n_hidden_layers, param=parameterization, perturb=perturbation, multiplier_mup = self.multiplier_mup)
    # corrected SP initialization and bias init from 13.11.24
    with torch.no_grad():
      for i in range(self.n_hidden_layers+1):
        # nn.init.kaiming_normal_(self.layers[i]['linear'].weight, mode='fan_in', nonlinearity='relu')
        fan_in = self.layers[i]['linear'].weight.size()[1]
        init_scaling = float(1/fan_in) if i == self.n_hidden_layers and parameterization=='mup' else float(1/np.sqrt(fan_in)) #mup has last layer smaller than rest.
        if 'largeinput' in parameterization and i==0:
          init_scaling = 1
        if parameterization == 'llm':
          # just width-independent scale -> way too large, but with AdamW and normlayer this should not matter
          init_scaling = 1 if i < self.n_hidden_layers else float(1/np.sqrt(fan_in))
          init_var = 0.02**2 if i < self.n_hidden_layers else 1
          
        self.layers[i]['linear'].weight.data.normal_(mean=0, std=float(np.sqrt(init_var)) * init_scaling)
        
        if use_bias: # INIT BIASES IN MUP JUST LIKE SP... SHOULD BE CORRECTED!
          if parameterization=='mup' or large_bias_init:
            bound = 1 if fan_in > 0 else 0 #fan_in(bias) = 1!
          else:
            # copied from torch.Linear: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            bound = float(1 / np.sqrt(fan_in)) if fan_in > 0 else 0
          init.uniform_(self.layers[i]['linear'].bias, -bound, bound)

        if self.initvarnorm and i==0 and (self.multipl_inout is not None):
          self.layers[i]['linear'].weight.data /= self.multipl_inout[0]
      if parameterization == 'mup' and self.multiplier_mup:
        for i in range(self.n_hidden_layers+1):
          self.layers[i]['linear'].weight.data.normal_(mean=0, std=float(np.sqrt(init_var)) * float(1/np.sqrt(hidden_size)))
        if self.multipl_inout is not None:
          self.multipl_inout = [self.multipl_inout[0]*(hidden_size**(1/2)), self.multipl_inout[-1]*(hidden_size**(-1/2))]
        else:
          self.multipl_inout = [(hidden_size**(1/2)), (hidden_size**(-1/2))]
          
    if self.ll_zero_init:
      nn.init.zeros_(self.layers[self.n_hidden_layers]['linear'].weight)
      if use_bias:
        nn.init.zeros_(self.layers[self.n_hidden_layers]['linear'].bias)
    
    
  def forward(self, x, out_layer=None):
    if self.flat_indim is None:
      y = x
    else:
      y=x.view(-1,self.flat_indim)
    for i, layer in enumerate(self.layers):
      if (i == 0) and (self.multipl_inout is not None):
        y_temp = self.multipl_inout[0] * layer['linear'](y)    
      elif (i==len(self.layers)-1) and (self.multipl_inout is not None):
        y_temp = self.multipl_inout[-1] * layer['linear'](y)
      else:
        y_temp = layer['linear'](y)
      if self.p_dropout is not None and i<self.n_hidden_layers: y_temp=layer['dropout'](y_temp)
      if layer['activation'] is not None:
        y_temp = layer['activation'](y_temp)
      if not self.res_connections or i in [0,len(self.layers)-1]:
        y = y_temp
      else:
        y = y + y_temp
      if out_layer is not None and i == out_layer:
        return y
    return y
  


  def get_bcd(self,param = None,perturb=None,variant=None, multiplier_mup = None):
    """For each trainable parameter, returns initialization, learning rate and perturbation exponent corresponding to the pre-specified parameterization."""
    # first layer, bias and bn always input,
    # last layer always output
    # conv hidden
    # shortcut.0=conv
    # shortcut.1=bn
    if param is None:
        param = self.parameterization
    if perturb is None:
        perturb = self.perturbation
    if variant is None:
        variant = self.variant
    if multiplier_mup is None:
        multiplier_mup = self.multiplier_mup

    # scalings for input-, hidden- and output-like params
    bl, cl, dl, d = get_bcd(L=2, param=param, perturb=perturb,variant=variant, multiplier_mup = multiplier_mup)
    bls, cls, dls = [],[],[]
    names = [name for name, param in self.named_parameters()]
    num_params = len(names)
    for name, param in self.named_parameters():
      if fnmatch.fnmatch(name,'linear_1.*') or fnmatch.fnmatch(name,'*.bias'):
        bls.append(bl[0])
        if fnmatch.fnmatch(name, names[-1].split('.',2)[0]+'.bias'):
          cls.append(0)
          dls.append(0)
        else:
          cls.append(cl[0])
          dls.append(dl[0])
      elif fnmatch.fnmatch(name, names[-1].split('.',2)[0]+'.weight'):
        bls.append(bl[-1])
        cls.append(cl[-1])
        dls.append(dl[-1])
      else:
        bls.append(bl[1])
        cls.append(cl[1])
        dls.append(dl[1])
    return bls, cls, dls, d







class RMSNorm(nn.Module):
    def __init__(self, dim, epsilon=1e-8):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))
        #self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        # Compute RMS of input
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        # Normalize and apply learnable scaling and shifting parameters
        return self.gamma * (x / rms)# + self.beta


class Scale_Corrected_MLP(torch.nn.Module):
  """
  Implements MLPs specifying LR and RHO parameterization. Initializes the MLP according to parameterization. Allows weight multipliers.
  """
  
  def __init__(self,
               in_size: int,
               out_size: int,
               hidden_size: int,
               n_hidden_layers: int,
               use_bias: bool = False,
               activation = F.relu,
               out_activation = None,
               res_connections = False,
               parameterization: str = 'mup',
               perturbation: str = 'mpp',
               init_var = 2,
               flat_indim: int = None,
               multipl_inout=None,# list of multipliers, so far only idx 0 and -1 supported
               large_bias_init = False,
               ll_zero_init = False,
               initvarnorm = False,
               multiplier_mup = False,
               norm_layer = None,# 'rms_hidden', # rms_hidden: not after W^L+1 (more common)
               out_norm_layer = None, #'layernorm': after W^L+1
               norm_eps = None,
               p_dropout = None):
    super().__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.hidden_size = hidden_size
    self.n_hidden_layers = n_hidden_layers
    self.n_layers = n_hidden_layers + 1
    self.use_bias = use_bias
    self.activation = activation
    self.out_activation = out_activation
    self.res_connections = res_connections
    self.parameterization = parameterization
    self.perturbation = perturbation
    self.init_var = init_var
    self.flat_indim = flat_indim
    self.multipl_inout = multipl_inout
    self.ll_zero_init = ll_zero_init
    self.large_bias_init = large_bias_init
    self.initvarnorm = initvarnorm
    self.multiplier_mup = multiplier_mup
    self.p_dropout = p_dropout
    self.norm_layer=norm_layer
    self.out_norm_layer = out_norm_layer
    self.norm_eps = norm_eps
    
    in_sizes = [self.in_size] + [self.hidden_size] * self.n_hidden_layers
    out_sizes = [self.hidden_size] * self.n_hidden_layers + [self.out_size]
    
    self.layers = []
    for i in range(self.n_hidden_layers+1):
      layer = {}
      layer['linear'] = torch.nn.Linear(in_sizes[i], out_sizes[i], bias=self.use_bias)
      self.add_module(f"linear_{i+1}", layer['linear'])
      if self.p_dropout is not None and i < self.n_hidden_layers:
        layer['dropout'] = torch.nn.Dropout(p=self.p_dropout)
        self.add_module(f"dropout_{i+1}", layer['dropout'])
      layer['activation'] = self.activation if i < self.n_hidden_layers  else self.out_activation
      if self.norm_layer == 'rms':
        layer['norm_layer'] = RMSNorm([out_sizes[i]]) if norm_eps is None else RMSNorm([out_sizes[i]],eps=norm_eps)
      elif self.norm_layer == 'rms_hidden':
        if i<self.n_hidden_layers:
          layer['norm_layer'] = RMSNorm([out_sizes[i]]) if norm_eps is None else RMSNorm([out_sizes[i]],eps=norm_eps)
        else:
          layer['norm_layer'] = None
      elif self.norm_layer is None:
        layer['norm_layer'] = None
      else:
        raise NotImplementedError()
      if i == self.n_hidden_layers and self.out_norm_layer == 'layernorm':
        layer['norm_layer'] = nn.LayerNorm(out_sizes[i]) if norm_eps is None else nn.LayerNorm(out_sizes[i], eps=norm_eps)
      self.layers.append(layer)
      
    bl, cl, dl, d = get_bcd(L=self.n_hidden_layers, param=parameterization, perturb=perturbation, multiplier_mup = self.multiplier_mup)
    # corrected SP initialization and bias init from 13.11.24
    with torch.no_grad():
      for i in range(self.n_hidden_layers+1):
        # nn.init.kaiming_normal_(self.layers[i]['linear'].weight, mode='fan_in', nonlinearity='relu')
        fan_in = self.layers[i]['linear'].weight.size()[1]
        init_scaling = float(1/fan_in) if i == self.n_hidden_layers and parameterization=='mup' else float(1/np.sqrt(fan_in)) #mup has last layer smaller than rest.
        if parameterization == 'llm':
          # just width-independent scale -> way too large, but with AdamW and normlayer this should not matter
          init_scaling = 1 if i < self.n_hidden_layers else float(1/np.sqrt(fan_in))
          init_var = 0.02**2 if i < self.n_hidden_layers else 1
        if parameterization == 'sp_largeinput' and i==0:
          init_scaling = 1

        self.layers[i]['linear'].weight.data.normal_(mean=0, std=float(np.sqrt(init_var)) * init_scaling)
        
        if use_bias: # INIT BIASES IN MUP JUST LIKE SP... SHOULD BE CORRECTED!
          if parameterization=='mup' or large_bias_init:
            bound = 1 if fan_in > 0 else 0 #fan_in(bias) = 1!
          else:
            # copied from torch.Linear: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            bound = float(1 / np.sqrt(fan_in)) if fan_in > 0 else 0
          init.uniform_(self.layers[i]['linear'].bias, -bound, bound)

        if self.initvarnorm and i==0 and (self.multipl_inout is not None):
          self.layers[i]['linear'].weight.data /= self.multipl_inout[0]
      if parameterization == 'mup' and self.multiplier_mup:
        for i in range(self.n_hidden_layers+1):
          self.layers[i]['linear'].weight.data.normal_(mean=0, std=float(np.sqrt(init_var)) * float(1/np.sqrt(hidden_size)))
        if self.multipl_inout is not None:
          self.multipl_inout = [self.multipl_inout[0]*(hidden_size**(1/2)), self.multipl_inout[-1]*(hidden_size**(-1/2))]
    if self.ll_zero_init:
      nn.init.zeros_(self.layers[self.n_hidden_layers]['linear'].weight)
      if use_bias:
        nn.init.zeros_(self.layers[self.n_hidden_layers]['linear'].bias)
    
    
  def forward(self, x, out_layer=None):
    if self.flat_indim is None:
      y = x
    else:
      y=x.view(-1,self.flat_indim)
    for i, layer in enumerate(self.layers):
      if (i == 0) and (self.multipl_inout is not None):
        y_temp = self.multipl_inout[0] * layer['linear'](y)    
      elif (i==len(self.layers)-1) and (self.multipl_inout is not None):
        y_temp = self.multipl_inout[-1] * layer['linear'](y)
      else:
        y_temp = layer['linear'](y)
      if self.p_dropout is not None and i<self.n_hidden_layers: y_temp=layer['dropout'](y_temp)
      if layer['activation'] is not None:
        y_temp = layer['activation'](y_temp)
      if layer['norm_layer'] is not None:
        y_temp = layer['norm_layer'](y_temp)
      if not self.res_connections or i in [0,len(self.layers)-1]:
        y = y_temp
      else:
        y = y + y_temp
      if out_layer is not None and i == out_layer:
        return y
    return y
  


  def get_bcd(self,param = None,perturb=None,variant=None, multiplier_mup = None):
    """For each trainable parameter, returns initialization, learning rate and perturbation exponent corresponding to the pre-specified parameterization."""
    # first layer, bias and bn always input,
    # last layer always output
    # conv hidden
    # shortcut.0=conv
    # shortcut.1=bn
    if param is None:
        param = self.parameterization
    if perturb is None:
        perturb = self.perturbation
    if variant is None:
        variant = self.variant
    if multiplier_mup is None:
        multiplier_mup = self.multiplier_mup

    # scalings for input-, hidden- and output-like params
    bl, cl, dl, d = get_bcd(L=2, param=param, perturb=perturb,variant=variant, multiplier_mup = multiplier_mup)
    bls, cls, dls = [],[],[]
    names = [name for name, param in self.named_parameters()]
    num_params = len(names)
    for name, param in self.named_parameters():
      if fnmatch.fnmatch(name,'linear_1.*') or fnmatch.fnmatch(name,'*.bias'):
        bls.append(bl[0])
        if fnmatch.fnmatch(name, names[-1].split('.',2)[0]+'.bias'):
          cls.append(0)
          dls.append(0)
        else:
          cls.append(cl[0])
          dls.append(dl[0])
      elif fnmatch.fnmatch(name, names[-1].split('.',2)[0]+'.weight'):
        bls.append(bl[-1])
        cls.append(cl[-1])
        dls.append(dl[-1])
      else:
        bls.append(bl[1])
        cls.append(cl[1])
        dls.append(dl[1])
    return bls, cls, dls, d
