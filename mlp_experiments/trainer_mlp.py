"""
Class for training an MLP with pre-specified learning rate and perturbation scaling parameterization (as defined in utils.mlp), with many optional evaluation metrics.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from absl import logging
from tqdm import tqdm

# from pyhessian import hessian
from utils import get_bcd, get_filename_from_args, compute_accuracy, inner_prod, find
from utils.eval import compute_feature_sing_vals, Hessian_eval
from mlp_experiments.utils.sam import SAM
from torch.linalg import matrix_norm, vector_norm
from utils.mlp import MLP

if torch.cuda.device_count()>0:
    torch.set_default_device('cuda')

# def hessian(x,y,dataloader=0,data=0,cuda=0):
#     return None

class MLPTrainer(object):
    """
    Class for training an MLP with pre-specified learning rate and perturbation scaling parameterization (as defined in utils.mlp), with many optional evaluation metrics.
    """
    
    def __init__(self,
                 model,
                 train_dataloader,
                 eval_dataloader,
                 lr: float,
                 rho: float = None,
                 optim_algo = 'SGD',
                 weight_decay = 0,
                 momentum = 0,
                 eps_adam = 1e-8,
                 scheduler = None,
                 exp_name: str = 'unknown',
                 device = None,
                 loss_fn = F.mse_loss,
                 classification: bool = False, # return_gradnorm_iter: int = 0, only returns nan
                 width_indep_gradnorms = False, # scale all gradnorm contributions to Theta(1)
                 gn0 = None, # list      set gradnormcontrib. from these layers to 0
                 extended_eval_iter: int = 0, # for extended eval: everything beyond train/val acc.
                 feature_rank_iter = None, # list of iter for which feature rank eval
                 eval_iter = 0, # eval train/val acc every self.eval_iter batches, not every epoch
                 save_best_file = None, #string to save best model to
                 save_final_file = None,
                 final_eval = False,
                 spectral_mup = False,
                 small_input_mup = False,
                 only_track_norms = False,
                 del_last_step = False,
                 tag_dict = {}):
        super().__init__()
        self.model = model
        if not del_last_step:
            self.init_model = copy.deepcopy(model) if extended_eval_iter > 0 or final_eval else None
        else:
            self.init_model = copy.deepcopy(model) if extended_eval_iter > 0 or final_eval else None
        self.widemodel = None
        # self.delta_model = copy.deepcopy(model)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        assert optim_algo in ['SGD', 'SAM-SGD', 'SAM-SGD-LL', 'LL-SGD', 'ADAM', 'LL-ADAM', 'NF-ADAM', 'NF-SGD'], "Unknown optim_algo"
        self.optim_algo = optim_algo
        self.lr = lr
        self.rho = rho
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.eps_adam = eps_adam
        self.scheduler = scheduler
        self.exp_name = exp_name
        self.device = device
        self.loss_fn = loss_fn
        self.classification = classification
        self.width_indep_gradnorms = width_indep_gradnorms
        self.gn0 = None if gn0 == ['None'] else gn0
        self.extended_eval_iter = extended_eval_iter
        self.feature_rank_iter = feature_rank_iter
        self.eval_iter = eval_iter
        self.final_eval = final_eval
        self.only_track_norms = only_track_norms
        self.del_last_step = del_last_step
        self.iter = 0 #counts the num_batches used for training
        self.epoch = 0
        self.total_time = 0
        self.total_eval_time = 0
        self.tag_dict = tag_dict
        self.save_best_file = save_best_file
        self.save_final_file = save_final_file
        self.best_valacc = 0.1
        self.spectral_mup = spectral_mup
        self.small_input_mup = small_input_mup
        self.eval_inputs_train, self.eval_targets_train = None, None
        self.name = f"{exp_name}/{get_filename_from_args(tag_dict)}"
        
        
        #############################  LAYERWISE LR AND RHO SCALING  #############################

        bl, cl, dl, d = get_bcd(L=self.model.n_hidden_layers,
                             param=self.model.parameterization,
                             perturb=self.model.perturbation,
                             multiplier_mup=self.model.multiplier_mup)
        
        width = float(model.hidden_size)
        if width_indep_gradnorms:
            if model.parameterization != 'mup':
                raise NotImplementedError('Width indep. gradnorm scaling is only implemented for MUP.')
            # 1/2-c_nabla, 1-c_nabla, 1/2
            gradnorm_dl = [0 for p in dl]
            gradnorm_dl[0] = -1/2
            gradnorm_dl[-1] = 1/2
            gradnorm_scaling = [width**(-p) for p in gradnorm_dl]# * (256**(p)) for base width
            if self.gn0 is not None:
                if isinstance(gn0,list) and len(gn0)==1:
                    gn0 = gn0[0]
                if isinstance(gn0,str):
                    gn0 = list(gn0)
                for thischar in ['[',']',',']:
                    try:
                        gn0.remove(thischar)
                    except:
                        continue
                self.gn0 = [int(thisidx) for thisidx in gn0]
                for thisidx in self.gn0:
                    gradnorm_scaling[thisidx] = 0
        else:
            gradnorm_scaling = [width ** (-p) for p in dl]
        
        param_groups = []
        names = [nam for nam, _ in self.model.named_parameters()]
        last_layer_name = names[-1].split('.',2)[0]
        first_layer_name = names[0].split('.',2)[0]
        for i, (name, params) in enumerate(self.model.named_parameters()):
            if optim_algo in  ['LL-SGD', 'LL-ADAM']:
                lr_mult = 0 if (last_layer_name not in name) else 1
            elif optim_algo in ['NF-SGD', 'NF-ADAM']:
                lr_mult = 0 if (first_layer_name in name) else 1
            else:
                lr_mult = 1
            if spectral_mup:
                if len(params.size()) == 2:
                    fan_out, fan_in = params.size()
                elif len(params.size()) == 1:
                    fan_out, fan_in = params.size()[0], 1
                lr_mult *= fan_out/fan_in
                if self.small_input_mup and i==0:
                    lr_mult /= np.sqrt(fan_out)
                rho_mult = 0 #TODO: implement for SAM
                if 'SAM' in optim_algo: raise ValueError('Spectral mup only implemented for SGD.')
                if self.model.multiplier_mup:
                    lr_mult /= lr_mult if lr_mult!=0 else 0
                param_groups.append(
                        {"params": params, 
                        "lr": lr * lr_mult,
                        "rho": rho * width ** (-d) * rho_mult,
                        "weight_decay": weight_decay / lr_mult if lr_mult>0 else 0, # scale inversely to lr due to PyTorch's coupling of lr*wd.
                        "perturb": 'SAM' in optim_algo,
                        "name": name}
                    )
            else:
                if self.model.use_bias:
                    if 'bias' in name:
                        fan_out = params.size()[0]
                        lr_mult *= fan_out
                    else:
                        if 'linear_1' in name:
                            lr_mult *= width **(-cl[0])
                            rho_mult = width **(-dl[0])
                            if self.small_input_mup:
                                lr_mult /= np.sqrt(width)
                        elif last_layer_name in name:
                            lr_mult *= width ** (-cl[-1])
                            rho_mult = width **(-dl[-1])
                        else:
                            lr_mult *= width ** (-cl[1])
                            rho_mult = width **(-dl[1])
                    if self.model.multiplier_mup:
                        lr_mult /= lr_mult if lr_mult!=0 else 0
                    param_groups.append(
                        {"params": params, 
                        "lr": lr * lr_mult,
                        "rho": rho * width ** (-d) * rho_mult,
                        "weight_decay": weight_decay / lr_mult if lr_mult>0 else 0, # scale inversely to lr due to PyTorch's coupling of lr*wd.
                        "perturb": 'SAM' in optim_algo,
                        "name": name}
                    )
                else:# like old implementation
                    if 'ADAM' in optim_algo: # SAM-ADAM not implemented
                        if len(params.size()) == 2:
                            fan_out, fan_in = params.size()
                        elif len(params.size()) == 1:
                            fan_out, fan_in = params.size()[0], 1
                        if 'mup' in self.model.parameterization:
                            lr_mult *= 1/fan_in
                            cl[i]=0
                        if self.model.parameterization == 'mup_spllit_largelr' and (last_layer_name in name): # ensure no normalization layer after W^{L+1}?
                            lr_mult *= np.sqrt(fan_in)
                    else:
                        if self.model.multiplier_mup:
                            lr_mult /= lr_mult if lr_mult!=0 else 0
                            cl[i]=0
                            dl[i]=0
                            d=0
                        if self.small_input_mup and i==0:
                            lr_mult /= np.sqrt(width)
                    param_groups.append(
                        {"params": params, 
                        "lr": lr * width ** (-cl[i]) * lr_mult,
                        "rho": rho * width ** (-d) * (width ** (-dl[i])),
                        "weight_decay": weight_decay * width ** (cl[i]) / lr_mult if lr_mult>0 else 0, # scale inversely to lr due to PyTorch's coupling of lr*wd.
                        "perturb": 'SAM' in optim_algo,
                        "name": f'layer{i}'}
                    )
            
        if optim_algo == 'SGD' or optim_algo == 'LL-SGD' or optim_algo == 'NF-SGD':
            self.optimizer = torch.optim.SGD(param_groups, 0.,momentum = self.momentum,weight_decay=self.weight_decay)
        elif optim_algo == 'ADAM' or optim_algo == 'LL-ADAM' or optim_algo == 'NF-ADAM':
            # if self.model.parameterization == 'mup':
            #     raise NotImplementedError('MUP not yet implemented for ADAM.')
            self.optimizer = torch.optim.AdamW(param_groups, lr=0, weight_decay=self.weight_decay,eps=self.eps_adam)
        elif optim_algo == 'SAM-SGD':
            self.optimizer = SAM(param_groups, torch.optim.SGD, gradnorm_scaling=gradnorm_scaling,momentum = self.momentum,weight_decay=self.weight_decay)
        elif optim_algo == 'SAM-SGD-LL':
            for i, group in enumerate(param_groups):
                if i < len(param_groups) - 1:
                    group['perturb'] = False
            self.optimizer = SAM(param_groups, torch.optim.SGD, gradnorm_scaling=gradnorm_scaling,momentum = self.momentum,weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
            
        
        #############################  INITIALIZE OPTIONAL EVALUATIONS  #############################

        logging.debug(self.optimizer)
        
        self.history = {
            'epoch': [],
            'iter': [],
            'Loss/optim': [],
            "Loss/train": [],
            "Loss/val": [],
            'traintime': [],
            'evaltime': []
        }
        for i in range(self.model.n_layers):
            if self.feature_rank_iter is not None or final_eval:
                self.history[f"feature_ranks_layer{i}"] = []
                self.history[f"singvalsum_layer{i}"] = []
            if self.extended_eval_iter > 0 or final_eval:
                self.history[f'activation_delta_layer{i}'] = []
                self.history[f'activation_del_layer{i}'] = []
                #self.history[f"Norm/spectral_norm/layer-{i}"] = []
                self.history[f"activation_l2_layer{i}"] = []
                self.history[f"activ0_layer{i}"] = []

            

        if self.extended_eval_iter > 0 or final_eval:
            self.history['gradnorms'] =[]
            self.history['W_spectral_norms'] =[]
            self.history['delta_W_spectral_norms'] =[]
            self.history['delta_W_frob_norms'] =[]
            self.history['ll_activation_norm'] =[]
            self.history['delta_W_x_norm'] =[] #last layer
            self.history['delta_W_x_norms'] =[] # all layers
            self.history['W_delta_x_norms'] =[] # all layers
            self.history['activationnorms'] =[] # all layers
            self.history['activationdeltanorms'] =[] # all layers
            self.history['W_init_spectral_norm'] =[]
            self.history['W_init_frob_norm'] =[]
            self.history['del_W_spectral_norms'] =[]
            self.history['del_W_frob_norms'] =[]
            self.history['del_W_x_norms'] =[]
            self.history['activ_align_init_samept'] =[]
            self.history['activ_align_init_otherpt'] =[]
            self.history['activ_align_init_samept_alll'] =[]
            self.history['activ_align_init_otherpt_alll'] =[]
            self.history['perturbnorms'] =[]

            if not self.only_track_norms:
                self.history['hessiannorm'] = []
                self.history['hessiannorms'] = []
                self.history['hessian_grad_alignment'] = []
                self.history['hessiangap'] = []
                self.history['hessiantrace'] = []
                self.history['l2diff2wide_train'] = []
                self.history['l2diff2wide_test'] = []

            if self.model.use_bias:
                self.history['b_norms'] = []
                self.history['delta_b_norms'] = []

    #############################  TRAINING WITH MANY OPTIONAL EVALUATIONS  #############################
    
    def record_stats(self, value, tag, time_step, epoch_count=True):
        """
        Record statistic 'value' in self.history with key 'tag'. If epoch_count==True, records the current epoch. Otherwise, records the current batch of training.
        """
        if tag in self.history:
            self.history[tag].append((time_step, value, epoch_count)) #added on 5.11.24
            if epoch_count:
                if len(self.history['epoch']) == 0 or time_step > self.history['epoch'][-1]:
                    self.history['epoch'].append(time_step)
            else:
                if len(self.history['iter']) == 0 or time_step > self.history['iter'][-1]:
                    self.history['iter'].append(time_step)
    
    
    def first_step_wrapper(self, only_ll = False):
        """
        Performs SAM's perturbation step with optional extended evaluations.
        """
        if self.extended_eval_iter>0:
            if (self.iter % self.extended_eval_iter==0) or self.iter < 20:
                perturbnorms = self.optimizer.first_step(zero_grad=True, return_perturbnorm = True)
                if only_ll: perturbnorms=perturbnorms[-1]
                self.record_stats(perturbnorms, 'perturbnorms', self.iter, epoch_count=False)
            else: self.optimizer.first_step(zero_grad=True)
        else:
            self.optimizer.first_step(zero_grad=True)
    
    
    def _train_one_epoch(self):
        """Train for one epoch with evaluations."""
        total_loss = 0
        epoch_time = 0
        eval_time = 0
            
        for inputs, targets in self.train_dataloader:
            if self.del_last_step and (self.extended_eval_iter>0) and ((self.iter<20) or (self.iter % self.extended_eval_iter == 0)): #TODO: this is very inefficient and does not work for final_eval
                del self.init_model
                self.init_model = copy.deepcopy(self.model)
            if self.classification:
                inputs = inputs.float() # for softmax
            if self.device is not None:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            # if self.random_labels:
            #     targets = torch.randint(0, 10, targets.size(), device=targets.device)
            if self.iter == 0 and (self.extended_eval_iter > 0 or self.final_eval):
                self.eval_inputs_train, self.eval_targets_train = inputs, targets
                self.init_activations = self.init_model(self.eval_inputs_train, out_layer=self.model.n_hidden_layers-1)
                self.activation_dict = {0: self.init_activations, 1: None, 10: None, 100: None, -1: self.init_activations}
                self.init_activations_alllayers = []
                for i_l in range(self.model.n_hidden_layers+1):
                    activations = self.eval_inputs_train.view(-1,self.model.flat_indim) if i_l==0 else self.model(self.eval_inputs_train, out_layer=i_l-1)
                    self.init_activations_alllayers.append(activations)
                keychain = '_'.join([f'{it}' for it in self.activation_dict.keys()])
                self.history[f"activ_align_past{keychain}"] =[]
                mat_norms, mat_norms_fro = [], []
                for i_l, layer in enumerate(self.model.layers):
                    try:
                        mat_norm = float(matrix_norm(layer['linear'].weight, ord=2).detach().cpu().numpy())
                    except:
                        mat_norm = np.nan
                    mat_norms.append(mat_norm)
                    try:
                        mat_norm_fro = float(matrix_norm(layer['linear'].weight, ord='fro').detach().cpu().numpy())
                    except:
                        mat_norm_fro = np.nan
                    mat_norms_fro.append(mat_norm_fro)

                self.history["W_init_spectral_norm"] = [mat_norms]
                self.history["W_init_frob_norm"] = [mat_norms_fro]
            temp_eval = time.time()
            if (self.eval_iter > 0) and ((self.iter<20) or (self.iter % self.eval_iter == 0)): # eval every self.eval_iter batches
                self.eval_loss_and_save_best_model(epoch_count=False, inputs=inputs, targets=targets)
            if (self.feature_rank_iter is not None) and (self.iter in self.feature_rank_iter):
                with torch.no_grad():
                    for i_l, layer in enumerate(self.model.layers):
                        feat_ranks, total, _ = compute_feature_sing_vals(self.train_dataloader, self.model, out_layer=i_l,
                                                                                      device=self.device,full_return=False, num_batches = 10)
                        self.record_stats(feat_ranks, tag=f"feature_ranks_layer{i_l}", time_step=self.iter, epoch_count=False)
                        self.record_stats(total, tag=f"singvalsum_layer{i_l}", time_step=self.iter, epoch_count=False)

            temp_time = time.time()
            eval_time += temp_time-temp_eval
            self.model.train()
            # if self.classification:
            #     inputs = inputs.float() # for softmax
            # if self.device is not None:
            #     inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.optim_algo in ['SGD', 'LL-SGD', 'ADAM', 'LL-ADAM', 'NF-SGD', 'NF-ADAM']:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
            elif 'SAM' in self.optim_algo:
                # first forward-backward pass
                loss = self.loss_fn(self.model(inputs),targets)
                loss.backward()
                self.first_step_wrapper()
                # second forward-backward pass
                self.loss_fn(self.model(inputs),targets).backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                raise NotImplementedError
            if self.scheduler is not None:
                self.scheduler.step()
            if self.eval_iter==0 and epoch_time==0:
                self.record_stats(loss.detach().cpu().numpy(), tag='Loss/optim',time_step= self.epoch)#first batch of epoch
            epoch_time += time.time()-temp_time
            temp_eval = time.time()
            # more evals
            if (self.extended_eval_iter>0) and ((self.iter<20) or (self.iter % self.extended_eval_iter == 0)):
                self.extended_eval(inputs, targets)
            
            if (self.extended_eval_iter > 0) and ((self.iter<20) or (self.iter % self.extended_eval_iter == 0)):
                for i_l, layer in enumerate(self.model.layers):
                    outmodel = self.model(self.eval_inputs_train,out_layer = i_l)
                    outinitmodel = self.init_model(self.eval_inputs_train,out_layer = i_l)
                    self.record_stats(np.linalg.norm((outinitmodel - outmodel).detach().cpu().numpy(),axis = 1).mean(),
                                      f'activation_delta_layer{i_l}',self.iter, epoch_count=False)
                    # outlastmodel = self.laststep_model(inputs,out_layer = i_l)
                    # self.record_stats(np.linalg.norm((outlastmodel - outmodel).detach().cpu().numpy(),axis = 1).mean(),
                    #                   f'activation_del_layer{i_l}',self.iter, epoch_count=False)
            eval_time += time.time()-temp_eval
            self.iter += 1
            
        return epoch_time, eval_time
    
    
    def train(self, n_epochs, resample_data=False):
        """Train for n_epoch, potentially with final evaluation."""
        if self.scheduler == 'cos':
            if self.optim_algo in ['SGD', 'LL-SGD', 'ADAM', 'LL-ADAM', 'NF-SGD', 'NF-ADAM']:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_epochs*len(self.train_dataloader))
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer.base_optimizer, T_max=n_epochs*len(self.train_dataloader))
        for epoch in tqdm(range(n_epochs)):
            self.epoch = epoch
            temp_eval = time.time()
            if (self.eval_iter == 0): # basic eval after every epoch
                self.eval_loss_and_save_best_model()

            self.total_eval_time += time.time()-temp_eval
            epoch_time, epoch_eval_time = self._train_one_epoch()
            self.total_time += epoch_time
            self.total_eval_time += epoch_eval_time
            self.record_stats(self.total_time, 'traintime', self.epoch)
            self.record_stats(self.total_eval_time, 'evaltime',self.epoch)
            torch.cuda.empty_cache()
            if resample_data:
                self.train_dataloader.dataset.resample_data()
        self.epoch += 1
        self.eval_loss_and_save_best_model()
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        if self.save_best_file is not None:
            try:
                torch.save(self.wrapdict, './checkpoint/'+self.save_best_file+'.pth')
            except:
                for icut in range(30):
                    try:
                        torch.save(self.wrapdict, './checkpoint/'+self.save_best_file[icut:-icut]+'.pth')
                    except:
                        continue
        if self.save_final_file is not None:
            train_loss = self.evaluate_loss(dataloader=self.train_dataloader)
            val_loss = self.evaluate_loss()
            state = {
                'net': self.model.state_dict(),
                'trainacc': train_loss,
                'valacc': val_loss,
                'epoch': self.epoch,
                'iter': self.iter,
            }
                
            wrapdict={}
            wrapdict[(self.model.hidden_size,self.lr,self.rho)] = state
            try:
                torch.save(wrapdict, './checkpoint/'+self.save_final_file+'.pth')
            except:
                for icut in range(30):
                    try:
                        torch.save(wrapdict, './checkpoint/'+self.save_final_file[icut:-icut]+'.pth')
                    except:
                        continue
        if self.final_eval:
            self.final_extended_eval()
    
    
    def evaluate_loss(self, dataloader = None):
        """Compute loss or accuracy. If self.classification, compute accuracy. Default dataloader is self.eval_dataloader."""
        self.model.eval()
        if dataloader is None:
            dataloader = self.eval_dataloader
        if self.classification:
            avg_loss=compute_accuracy(self.model, dataloader, self.device).item()
            return avg_loss
        total_loss = 0
        with torch.no_grad():  
            for inputs, targets in dataloader:
                if self.device is not None:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss
    

    def eval_loss_and_save_best_model(self,epoch_count = True, inputs=None, targets=None):
        with torch.no_grad():
            train_loss = self.evaluate_loss(dataloader=self.train_dataloader)
            val_loss = self.evaluate_loss()
            if epoch_count:
                logging.debug(f"Epoch {self.epoch}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}")
                self.record_stats(train_loss, 'Loss/train', self.epoch)
                self.record_stats(val_loss, 'Loss/val', self.epoch)
            else:
                logging.debug(f"Iter {self.iter}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}")
                self.record_stats(train_loss, 'Loss/train', self.iter, epoch_count=False)
                self.record_stats(val_loss, 'Loss/val', self.iter, epoch_count=False)
                loss = self.loss_fn(self.model(inputs), targets)
                self.record_stats(loss.detach().cpu().numpy(),'Loss/optim',self.iter, epoch_count=False)
                
            if self.save_best_file is not None:
                if val_loss > self.best_valacc:
                    if epoch_count:
                        state = {
                                'net': self.model.state_dict(),
                                'trainacc': train_loss,
                                'valacc': val_loss,
                                'epoch': self.epoch,
                            }
                    else:
                        state = {
                            'net': self.model.state_dict(),
                            'trainacc': train_loss,
                            'valacc': val_loss,
                            'iter': self.iter,
                        }
                        
                    self.wrapdict={}
                    self.wrapdict[(self.model.hidden_size,self.lr,self.rho)] = state
                    
                    self.best_valacc = val_loss
    
    def extended_eval(self,inputs, targets):
        mat_norms, delta_mat_norms, delta_mat_norms_fro, b_norms, delta_b_norms, del_mat_norms, del_mat_norms_fro, = [], [], [], [], [], [], []
        
        # for i_l, layer in enumerate(self.model.layers):
        #     self.delta_model.layers[i_l] = layer - self.init_model.layers[i_l]
        
        for i_l, layer in enumerate(self.model.layers):
            try:
                mat_norm = float(matrix_norm(layer['linear'].weight, ord=2).detach().cpu().numpy())
            except:
                mat_norm = np.nan
            mat_norms.append(mat_norm)
            
            init_layer = self.init_model.layers[i_l]
            try:
                delta_mat_norm = float(matrix_norm(layer['linear'].weight - init_layer['linear'].weight, ord=2).detach().cpu().numpy())
            except:
                delta_mat_norm = np.nan
            try:
                delta_mat_norm_fro = float(matrix_norm(layer['linear'].weight - init_layer['linear'].weight, ord='fro').detach().cpu().numpy())
            except:
                delta_mat_norm_fro = np.nan
            delta_mat_norms.append(delta_mat_norm)
            delta_mat_norms_fro.append(delta_mat_norm_fro)

            #laststep_layer = self.laststep_model.layers[i_l]
            # try:
            #     del_mat_norm = float(matrix_norm(layer['linear'].weight - laststep_layer['linear'].weight, ord=2).detach().cpu().numpy())
            # except:
            #     del_mat_norm = np.nan
            # try:
            #     del_mat_norm_fro = float(matrix_norm(layer['linear'].weight - laststep_layer['linear'].weight, ord='fro').detach().cpu().numpy())
            # except:
            #     del_mat_norm_fro = np.nan
            # del_mat_norms.append(del_mat_norm)
            # del_mat_norms_fro.append(del_mat_norm_fro)

            if self.model.use_bias:
                bias_norm = vector_norm(layer['linear'].bias).detach().cpu().numpy()
                delta_bias_norm = vector_norm(layer['linear'].bias-init_layer['linear'].bias).detach().cpu().numpy()
                b_norms.append(bias_norm)
                delta_b_norms.append(delta_bias_norm)
            
            outmodel = self.model(self.eval_inputs_train,out_layer = i_l) # changed on 27.4. for evaluating ||Delta W x||
            outmodel = outmodel.detach().cpu().numpy()
            self.record_stats(np.linalg.norm(outmodel,axis = 1).mean(),f'activation_l2_layer{i_l}',self.iter,epoch_count=False)
            self.record_stats(np.mean((outmodel==0)),f'activ0_layer{i_l}',self.iter,epoch_count=False)
            
        ll_activations = self.model(self.eval_inputs_train, out_layer=self.model.n_hidden_layers-1)

        for act_iter in self.activation_dict:
            if self.iter == act_iter:
                self.activation_dict[act_iter] = ll_activations

        if self.model.use_bias:
            delta_W_x_norm = -999 #vector_norm(self.model.layers[-1]['linear'](ll_activations) - self.init_model.layers[-1]['linear'](ll_activations)).detach().cpu().numpy()  #(self.delta_model.layers[-1](ll_activations))
        else:
            delta_W_x_norm = vector_norm(self.model.layers[-1]['linear'](ll_activations) - self.init_model.layers[-1]['linear'](ll_activations)).detach().cpu().numpy()  #(self.delta_model.layers[-1](ll_activations))
        activ_alignment_init = inner_prod(ll_activations, self.init_activations).detach().cpu().numpy()
        activ_alignment_init_samept = np.trace(activ_alignment_init)/activ_alignment_init.shape[0]
        activ_alignment_init_otherpt = np.mean(activ_alignment_init[~np.eye(len(activ_alignment_init),dtype=bool)])
        activ_alignment_past = []
        for act_iter in self.activation_dict:
            try:
                activ_alignment_past.append(inner_prod(self.activation_dict[act_iter], ll_activations).mean().detach().cpu().numpy())
            except:
                activ_alignment_past.append(np.nan)
        self.record_stats(mat_norms, tag="W_spectral_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(delta_mat_norms, tag="delta_W_spectral_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(delta_mat_norms_fro, tag="delta_W_frob_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(del_mat_norms, tag="del_W_spectral_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(del_mat_norms_fro, tag="del_W_frob_norms", time_step=self.iter, epoch_count=False)
        if self.model.use_bias:
            self.record_stats(b_norms, tag="b_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(delta_b_norms, tag="delta_b_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(vector_norm(ll_activations).detach().cpu().numpy(), tag="ll_activation_norm", time_step=self.iter, epoch_count=False)
        self.record_stats(delta_W_x_norm, tag="delta_W_x_norm", time_step=self.iter, epoch_count=False)
        self.record_stats(activ_alignment_init_samept, tag="activ_align_init_samept", time_step=self.iter, epoch_count=False)
        self.record_stats(activ_alignment_init_otherpt, tag="activ_align_init_otherpt", time_step=self.iter, epoch_count=False)
        keychain = '_'.join([f'{it}' for it in self.activation_dict.keys()])
        self.record_stats(activ_alignment_past, tag=f"activ_align_past{keychain}", time_step=self.iter, epoch_count=False)
        self.activation_dict[-1] = ll_activations
        delta_W_x_norms, W_delta_x_norms, activationnorms, activationdeltanorms = [], [], [], []
        activ_alignment_init_samept_alllayers, activ_alignment_init_otherpt_alllayers = [],[]
        for i_l in range(self.model.n_hidden_layers+1):
            activations = self.eval_inputs_train.view(-1,self.model.flat_indim) if i_l==0 else self.model(self.eval_inputs_train, out_layer=i_l-1)
            init_activations = self.eval_inputs_train.view(-1,self.model.flat_indim) if i_l==0 else self.init_model(self.eval_inputs_train, out_layer=i_l-1)
            delta_W_x_norms.append(vector_norm(self.model.layers[i_l]['linear'](activations) - self.init_model.layers[i_l]['linear'](activations)).detach().cpu().numpy())
            W_delta_x_norms.append(vector_norm(self.init_model.layers[i_l]['linear'](activations)-self.init_model.layers[i_l]['linear'](init_activations)).detach().cpu().numpy())
            activationnorms.append(vector_norm(activations).detach().cpu().numpy())
            activationdeltanorms.append(vector_norm(activations-init_activations).detach().cpu().numpy())
            #del_W_x_norms.append(vector_norm(self.model.layers[i_l]['linear'](activations) - self.laststep_model.layers[i_l]['linear'](activations)).detach().cpu().numpy())
            activ_alignment_init = inner_prod(activations, self.init_activations_alllayers[i_l]).detach().cpu().numpy()
            activ_alignment_init_samept = np.trace(activ_alignment_init)/activ_alignment_init.shape[0]
            activ_alignment_init_otherpt = np.mean(activ_alignment_init[~np.eye(len(activ_alignment_init),dtype=bool)])
            activ_alignment_init_samept_alllayers.append(activ_alignment_init_samept)
            activ_alignment_init_otherpt_alllayers.append(activ_alignment_init_otherpt)
        self.record_stats(delta_W_x_norms, tag="delta_W_x_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(activationnorms, tag="activationnorms", time_step=self.iter, epoch_count=False)
        self.record_stats(W_delta_x_norms, tag="W_delta_x_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(activationdeltanorms, tag="activationdeltanorms", time_step=self.iter, epoch_count=False)
        #self.record_stats(del_W_x_norms, tag="del_W_x_norms", time_step=self.iter, epoch_count=False)
        self.record_stats(activ_alignment_init_samept_alllayers, tag="activ_align_init_samept_alll", time_step=self.iter, epoch_count=False)
        self.record_stats(activ_alignment_init_otherpt_alllayers, tag="activ_align_init_otherpt_alll", time_step=self.iter, epoch_count=False)

        
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(inputs),targets)
        loss.backward()
        grads=[]
        for _, param in self.model.named_parameters():
            grads.append(param.grad.detach().cpu().numpy())
        self.record_stats([np.linalg.norm(grad) for grad in grads],'gradnorms',self.iter, epoch_count=False)
        if not self.only_track_norms:
            Hess_eval=Hessian_eval(self.model,self.loss_fn, self.eval_inputs_train, self.eval_targets_train, max_iter=1000, diff_quot_stepsize=1e-4, eps_norm = 1e-4, to_cpu=False)
            spec_norm, grad_norm, grad_alignment, spec_vec, spec_norms = Hess_eval.eval()
            self.record_stats(spec_norm, 'hessiannorm', time_step=self.iter, epoch_count=False)
            self.record_stats(spec_norms, 'hessiannorms', time_step=self.iter, epoch_count=False)
            self.record_stats(grad_alignment, 'hessian_grad_alignment', time_step=self.iter, epoch_count=False)
            del Hess_eval
            # hessian_comp = hessian(self.model,self.loss_fn,data=(inputs,targets),cuda=True)
            # top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=2)
            # trace = np.mean(hessian_comp.trace())
            # self.record_stats(top_eigenvalues[0], 'hessiannorm',  self.iter, epoch_count=False)
            # self.record_stats(top_eigenvalues[0]-top_eigenvalues[1], 'hessiangap',  self.iter, epoch_count=False)
            # self.record_stats(trace, 'hessiantrace',  self.iter, epoch_count=False)
            # del hessian_comp
        


    

    def final_extended_eval(self):
        for inputs, targets in self.train_dataloader:
            if self.classification:
                inputs = inputs.float() # for softmax
            if self.device is not None:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.eval_inputs_train is None:
                self.eval_inputs_train, self.eval_targets_train = inputs, targets
            break
        if not self.only_track_norms:
            Hess_eval=Hessian_eval(self.model,self.loss_fn, self.eval_inputs_train, self.eval_targets_train, max_iter=1000, diff_quot_stepsize=1e-4, eps_norm = 1e-4, to_cpu=False)
            spec_norm, grad_norm, grad_alignment, spec_vec, spec_norms = Hess_eval.eval()
            self.record_stats(spec_norm, 'hessiannorm', self.epoch)
            self.record_stats(spec_norms, 'hessiannorms', self.epoch)
            self.record_stats(grad_alignment, 'hessian_grad_alignment', self.epoch)
            del Hess_eval
            #hessian_comp = hessian(self.model,self.loss_fn,dataloader=self.train_dataloader,cuda=True)
            #top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=2)
            #trace = np.mean(hessian_comp.trace())
            #self.record_stats(top_eigenvalues[0], 'hessiannorm', self.epoch)
            #self.record_stats(top_eigenvalues[0]-top_eigenvalues[1], 'hessiangap', self.epoch)
            #self.record_stats(trace, 'hessiantrace', self.epoch)
            #del hessian_comp
        with torch.no_grad():
            train_loss = self.evaluate_loss(dataloader=self.train_dataloader)
            val_loss = self.evaluate_loss()
            logging.debug(f"Epoch {self.epoch}, Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}")
            self.record_stats(train_loss, 'Loss/train', self.epoch)
            self.record_stats(val_loss, 'Loss/val', self.epoch)
            
            mat_norms, delta_mat_norms, delta_mat_norms_fro, b_norms, delta_b_norms, del_mat_norms, del_mat_norms_fro = [], [], [], [], [], [], []
            # for i_l, layer in enumerate(self.model.layers):
            #     self.delta_model.layers[i_l] = layer - self.init_model.layers[i_l]
            
            for i_l, layer in enumerate(self.model.layers):
                feat_ranks, total, _ = compute_feature_sing_vals(self.train_dataloader, self.model, out_layer=i_l,
                                                                            device=self.device,full_return=False, num_batches = 10)
                self.record_stats(feat_ranks, tag=f"feature_ranks_layer{i_l}", time_step=self.epoch)
                self.record_stats(total, tag=f"singvalsum_layer{i_l}", time_step=self.epoch)
                try:
                    mat_norm = float(matrix_norm(layer['linear'].weight, ord=2).detach().cpu().numpy())
                except:
                    mat_norm = np.nan
                mat_norms.append(mat_norm)
                
                init_layer = self.init_model.layers[i_l]
                try:
                    delta_mat_norm = float(matrix_norm(layer['linear'].weight - init_layer['linear'].weight, ord=2).detach().cpu().numpy())
                except:
                    delta_mat_norm = np.nan
                try:
                    delta_mat_norm_fro = float(matrix_norm(layer['linear'].weight - init_layer['linear'].weight, ord='fro').detach().cpu().numpy())
                except:
                    delta_mat_norm_fro = np.nan
                delta_mat_norms.append(delta_mat_norm)
                delta_mat_norms_fro.append(delta_mat_norm_fro)

                # laststep_layer = self.laststep_model.layers[i_l]
                # try:
                #     del_mat_norm = float(matrix_norm(layer['linear'].weight - laststep_layer['linear'].weight, ord=2).detach().cpu().numpy())
                # except:
                #     del_mat_norm = np.nan
                # try:
                #     del_mat_norm_fro = float(matrix_norm(layer['linear'].weight - laststep_layer['linear'].weight, ord='fro').detach().cpu().numpy())
                # except:
                #     del_mat_norm_fro = np.nan
                # del_mat_norms.append(del_mat_norm)
                # del_mat_norms_fro.append(del_mat_norm_fro)

                if self.model.use_bias:
                    bias_norm = vector_norm(layer['linear'].bias).detach().cpu().numpy()
                    delta_bias_norm = vector_norm(layer['linear'].bias-init_layer['linear'].bias).detach().cpu().numpy()
                    b_norms.append(bias_norm)
                    delta_b_norms.append(delta_bias_norm)
                outmodel = self.model(inputs,out_layer = i_l)
                outmodel = outmodel.detach().cpu().numpy()
                self.record_stats(np.linalg.norm(outmodel,axis = 1).mean(),f'activation_l2_layer{i_l}',self.iter,epoch_count=False)
                self.record_stats(np.mean((outmodel==0)),f'activ0_layer{i_l}',self.iter,epoch_count=False)
                
            ll_activations = self.model(self.eval_inputs_train, out_layer=self.model.n_hidden_layers-1)

            for act_iter in self.activation_dict:
                if self.iter == act_iter:
                    self.activation_dict[act_iter] = ll_activations

            delta_W_x_norm = vector_norm(self.model.layers[-1]['linear'](ll_activations) - self.init_model.layers[-1]['linear'](ll_activations)).detach().cpu().numpy()  #(self.delta_model.layers[-1](ll_activations))
            activ_alignment_init = inner_prod(ll_activations, self.init_activations).detach().cpu().numpy()
            activ_alignment_init_samept = np.trace(activ_alignment_init)/activ_alignment_init.shape[0]
            activ_alignment_init_otherpt = np.mean(activ_alignment_init[~np.eye(len(activ_alignment_init),dtype=bool)])
            activ_alignment_past = []
            for act_iter in self.activation_dict:
                try:
                    activ_alignment_past.append(inner_prod(self.activation_dict[act_iter], ll_activations).mean().detach().cpu().numpy())
                except:
                    activ_alignment_past.append(np.nan)
            self.record_stats(mat_norms, tag="W_spectral_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(delta_mat_norms, tag="delta_W_spectral_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(delta_mat_norms_fro, tag="delta_W_frob_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(del_mat_norms, tag="del_W_spectral_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(del_mat_norms_fro, tag="del_W_frob_norms", time_step=self.iter, epoch_count=False)
            if self.model.use_bias:
                self.record_stats(b_norms, tag="b_norms", time_step=self.iter, epoch_count=False)
                self.record_stats(delta_b_norms, tag="delta_b_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(vector_norm(ll_activations).detach().cpu().numpy(), tag="ll_activation_norm", time_step=self.iter, epoch_count=False)
            self.record_stats(delta_W_x_norm, tag="delta_W_x_norm", time_step=self.iter, epoch_count=False)
            self.record_stats(activ_alignment_init_samept, tag="activ_align_init_samept", time_step=self.iter, epoch_count=False)
            self.record_stats(activ_alignment_init_otherpt, tag="activ_align_init_otherpt", time_step=self.iter, epoch_count=False)
            keychain = '_'.join([f'{it}' for it in self.activation_dict.keys()])
            self.record_stats(activ_alignment_past, tag=f"activ_align_past{keychain}", time_step=self.iter, epoch_count=False)
            self.activation_dict[-1] = ll_activations
            delta_W_x_norms, W_delta_x_norms = [], []
            activ_alignment_init_samept_alllayers, activ_alignment_init_otherpt_alllayers = [],[]
            for i_l in range(self.model.n_hidden_layers+1):
                activations = self.eval_inputs_train.view(-1,self.model.flat_indim) if i_l==0 else self.model(self.eval_inputs_train, out_layer=i_l-1)
                delta_W_x_norms.append(vector_norm(self.model.layers[i_l]['linear'](activations) - self.init_model.layers[i_l]['linear'](activations)).detach().cpu().numpy())
                W_delta_x_norms.append(vector_norm(self.init_model.layers[i_l]['linear'](activations)-self.init_model.layers[i_l]['linear'](self.init_activations)).detach().cpu().numpy())
                #del_W_x_norms.append(vector_norm(self.model.layers[i_l]['linear'](activations) - self.laststep_model.layers[i_l]['linear'](activations)).detach().cpu().numpy())
                activ_alignment_init = inner_prod(activations, self.init_activations_alllayers[i_l]).detach().cpu().numpy()
                activ_alignment_init_samept = np.trace(activ_alignment_init)/activ_alignment_init.shape[0]
                activ_alignment_init_otherpt = np.mean(activ_alignment_init[~np.eye(len(activ_alignment_init),dtype=bool)])
                activ_alignment_init_samept_alllayers.append(activ_alignment_init_samept)
                activ_alignment_init_otherpt_alllayers.append(activ_alignment_init_otherpt)
            self.record_stats(delta_W_x_norms, tag="delta_W_x_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(W_delta_x_norms, tag="W_delta_x_norms", time_step=self.iter, epoch_count=False)
            #self.record_stats(del_W_x_norms, tag="del_W_x_norms", time_step=self.iter, epoch_count=False)
            self.record_stats(activ_alignment_init_samept_alllayers, tag="activ_align_init_samept_alll", time_step=self.iter, epoch_count=False)
            self.record_stats(activ_alignment_init_otherpt_alllayers, tag="activ_align_init_otherpt_alll", time_step=self.iter, epoch_count=False)
                
                
        running_loss = 0
        running_delta, running_del, running_perturbnorms = [[0 for i_l, layer in enumerate(self.model.layers)] for _ in range(3)]
        curr_state_dict = copy.deepcopy(self.model.state_dict())

        running_gradnorms = [0 for name, param in self.model.named_parameters()]
        for inputs,targets in self.train_dataloader:
            if self.classification:
                inputs = inputs.float() # for softmax
            if self.device is not None:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.load_state_dict(curr_state_dict)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(inputs),targets)
            loss.backward()
            grads=[]
            for name, param in self.model.named_parameters():
                grads.append(param.grad.detach().cpu().numpy())
            grad_norms = [np.linalg.norm(grad) for grad in grads]

            running_loss += loss.detach().cpu().numpy()
            running_gradnorms = [running_gradnorms[igrad] + grad_norms[igrad] for igrad in range(len(grad_norms))]
            if 'SAM' in self.optim_algo:
                try:
                    perturbnorms = self.optimizer.first_step(zero_grad=True, return_perturbnorm = True)
                    running_perturbnorms = [running_perturbnorms[i_l] + perturbnorms[i_l] for i_l in range(len(perturbnorms))]
                except:
                    print('ERROR: No perturbnorms in finaleval.')
            self.model.eval()
            for i_l, layer in enumerate(self.model.layers):
                outmodel = self.model(self.eval_inputs_train,out_layer = i_l)
                outinitmodel = self.init_model(self.eval_inputs_train,out_layer = i_l)
                #outlastmodel = self.laststep_model(inputs,out_layer = i_l)
                delta = np.linalg.norm((outinitmodel - outmodel).detach().cpu().numpy(),axis = 1).mean()
                running_delta[i_l] += delta
                #thisdel = np.linalg.norm((outlastmodel - outmodel).detach().cpu().numpy(),axis = 1).mean()
                #running_del[i_l] += thisdel
        self.record_stats([thisgn/len(self.train_dataloader) for thisgn in running_gradnorms], 'gradnorms', self.epoch)
        self.record_stats(running_loss/ len(self.train_dataloader), 'Loss/optim', self.epoch)
        self.record_stats([thispn/ len(self.train_dataloader) for thispn in running_perturbnorms], 'perturbnorms', self.epoch)
        for i_l, layer in enumerate(self.model.layers):
            self.record_stats(running_delta[i_l]/ len(self.train_dataloader),f'activation_delta_layer{i_l}',self.epoch)
            #self.record_stats(running_del[i_l]/ len(self.train_dataloader),f'activation_del_layer{i_l}',self.epoch)
        
        # load wide model and compute difference on test set:
        if self.widemodel is None and self.model.hidden_size < 16384:
            seed=self.exp_name.split('-seed=',10)[1].split('-',2)[0]
            checkpt_wide = None
            for checkpt_wide in find(f'finalmodel_mlp-with=16384-lr=' + self.exp_name.split('-lr=',10)[1].split('-seed=',2)[0]+f'-seed={seed}*','./checkpoint/'): break
            if checkpt_wide is not None:
                wide_model_wrap = torch.load(checkpt_wide)#'./checkpoint/'+f'finalmodel_mlp-with=16384-lr=' + self.exp_name.split('-lr=',10)[1].split('-sael',2)[0]+'-sael.pth')
                for widekey in wide_model_wrap: break
                self.wide_model = MLP(in_size=self.model.in_size,
                            out_size=self.model.out_size,
                            hidden_size=widekey[0],
                            n_hidden_layers=self.model.n_hidden_layers,
                            parameterization=self.model.parameterization,
                            perturbation=self.model.perturbation,
                            flat_indim = self.model.flat_indim,
                            multipl_inout=self.model.multipl_inout,
                            ll_zero_init=self.model.ll_zero_init,
                            initvarnorm=self.model.initvarnorm)
                self.wide_model.load_state_dict(wide_model_wrap[widekey]['net'])
                self.wide_model.eval()
                del wide_model_wrap
        if self.model.hidden_size < 16384 and checkpt_wide is not None and not self.only_track_norms:
            train_diff = np.linalg.norm(self.wide_model(inputs).detach().cpu().numpy()-self.model(inputs).detach().cpu().numpy(),axis=1).mean()
            self.record_stats(train_diff,'l2diff2wide_train',self.iter,epoch_count=False)
            total_diff = 0
            with torch.no_grad():  
                for test_inputs, test_targets in self.eval_dataloader:
                    if self.device is not None:
                        test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)
                    outputs = self.model(test_inputs).detach().cpu().numpy()
                    wide_outputs = self.wide_model(test_inputs).detach().cpu().numpy()
                    l2diff_wide = np.linalg.norm(outputs-wide_outputs,axis=1).mean()
                    total_diff += l2diff_wide
                avg_diff = total_diff / len(self.eval_dataloader)
                self.record_stats(avg_diff,'l2diff2wide_test',self.iter,epoch_count=False)
        else:
            self.record_stats(0,'l2diff2wide_train',self.iter,epoch_count=False)
            self.record_stats(0,'l2diff2wide_test',self.iter,epoch_count=False)

    
    
    def save_model(self, save_to):
        """Save current model state dict."""
        torch.save(self.model.state_dict(), f'{save_to}.pth')
        
    def visualize(self):
        """Pass."""
        pass