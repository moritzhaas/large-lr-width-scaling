import fnmatch
import io
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torch.linalg import vector_norm


def inner_prod(act1, act2):
    try:
        act1_norm = act1 / act1.norm(dim=1, keepdim=True)
        act2_norm = act2 / act2.norm(dim=1, keepdim=True)
        
        # return the matrix of scalar products between the rows of normalized a and b
        return torch.matmul(act1_norm, act2_norm.T)  # shape: (batch_size, batch_size)
        #WRONG: return (act1 @ act2.T)/(vector_norm(act1) * vector_norm(act2))
    except TypeError:
        return torch.nan

def get_filename_from_args(tag_dict, prefix=None, join=None):
    tags = []
    if prefix is not None:
        tags.append(str(prefix))
    for k, v in tag_dict.items():
        #6.1. commented out: if str(k) == 'nors' and not v: continue
        if isinstance(v,bool) and v:
            tags.append(str(k))
        elif v == 0:
            tags.append(str(k)+'=0')
        else:
            tags.append(str(k) + "=" + str(v))
    if join is None:
        return "--".join(tags)
    else:
        return join.join(tags)
    
    
def get_expname_from_args(args,prefix="compare",join='-'):
    tag_dict = {}
    for key in vars(args):
        if len(key)<=4:
            tag_dict[key] = np.round(vars(args)[key],6) if isinstance(vars(args)[key], float) else vars(args)[key]
        else:
            tag_dict[key[:2]+key[-2:]] = np.round(vars(args)[key],6) if isinstance(vars(args)[key], float) else vars(args)[key]
    tag_dict.pop('terd', None)
    tag_dict.pop('gpex',None)
    tag_dict.pop('daet',None)
    tag_dict.pop('prix',None)
    #tag_dict.pop('inrm',None)
    dellist = ['exer','ever','nuls','paus','lats']
    for key in dellist:
        if key in tag_dict.keys() and tag_dict[key] == 0:
            tag_dict.pop(key,None)
    if key in tag_dict.keys() and tag_dict['hpce'] == 'all':
        tag_dict.pop('hpce',None)
    for key in ['rhin','rhut','rhen','hilt','inlt']:
        if key in tag_dict.keys() and tag_dict[key] == 1.0:
            tag_dict.pop(key,None)
    if 'grng' in tag_dict.keys():
        if tag_dict['grng'] == 'relative': tag_dict['grng'] = 'rel'
        elif tag_dict['grng'] == 'original': tag_dict['grng'] = 'orig'
    return get_filename_from_args(tag_dict=tag_dict,prefix=prefix,join=join)

def get_next_hp_in_filename(filename, hp):
    # returns next_hp, this_value
    bool_hps = ['nors','llit','inms','inrm', 'rals', 'inls', 'usas','fial','sael']
    # TODO: keep list up to date and in order. save and load from file when exp_name is called?
    known_hps = ['with', 'lr', 'rho', 'nend', 'nell', 'paam', 'perb', 'opim', 'nors', 'llit', 'usas', 'inms', 'gn0', 'inrm', 'nhrs', 'bs', 'weay', 'moum', 'lang', 'lall', 'seed', 'laim', 'ever', 'fial', 'rals', 'inls', 'smze', 'sael']

    # if short_name:
    #     known_hps = ['with', 'lr', 'nend', 'nell', 'opim', 'usas', 'nhrs', 'bs', 'weay', 'moum', 'lang', 'lall', 'seed', 'laim', 'ever', 'fial', 'rals', 'inls', 'smze', 'sael']
    # else:
    #     known_hps = ['with', 'lr', 'rho', 'nend', 'nell', 'paam', 'perb', 'opim', 'nors', 'llit', 'usas', 'inms', 'gn0', 'inrm', 'nhrs', 'bs', 'weay', 'moum', 'lang', 'lall', 'seed', 'laim', 'ever', 'fial', 'rals', 'inls', 'smze', 'sael']
    try:
        hp_idx = np.where([hp==this_hp for this_hp in known_hps])[0][0]
    except IndexError:
        return None, None
    if hp_idx+1==len(known_hps):
        if filename.split(hp,2)[1][0]=='.': return None, True
        elif filename.split(hp,2)[1][:2]=='=0': return None, False
        else: raise ValueError(f'{hp} is last hp in {filename}')
    next_hp = known_hps[hp_idx+1]
    try:
        relevant = filename.split(hp,2)[1]
    except:
        return None, None
    if not next_hp in relevant:
        next_hp_idx = hp_idx + 2
        next_hp = known_hps[next_hp_idx]
        while next_hp not in relevant:
            try:
                next_hp_idx += 1
                next_hp = known_hps[next_hp_idx]
            except IndexError:
                raise ValueError(f'No known hp found after {hp} in {filename}')
    this_val = relevant.split('-'+next_hp,2)[0]
    if this_val == '': return next_hp, True
    else:
        this_val = this_val.split('=',2)[1]
        val_changed = True
        while val_changed:
            val_changed = False
            for this_hp in known_hps:
                if this_hp in this_val:
                    this_val = this_val.split('-'+this_hp,2)[0]
                    val_changed = True
        return next_hp, this_val

def shorten_stat_name(filenam):
    rough_hps = filenam.split('=',100)
    #print(rough_hps)
    hps=[thishp.split('-',10)[-1] for thishp in rough_hps]
    for hp in hps:
        if hp in ['rho','paam','perb','nors','llit','inms','gn0','inrm']: #irrelevant, as never varied
            next_hp, this_val = get_next_hp_in_filename(filenam, hp)
            #print(hp, this_val, next_hp)
            filenam=filenam.replace('-'+hp+'='+this_val+'-','-')
    #filenam=filenam.replace('-nors-','-')
    return filenam

def get_hps_from_filename(filename):
    hps={}
    known_hps = ['with', 'lr', 'rho', 'nend', 'nell', 'paam', 'perb', 'opim', 'nors', 'llit', 'usas', 'inms', 'gn0', 'inrm', 'nhrs', 'bs', 'weay', 'moum', 'lang', 'lall', 'seed', 'laim', 'ever', 'fial', 'rals', 'inls', 'smze', 'sael']
    for hp in known_hps:
        next_hp, this_val = get_next_hp_in_filename(filename, hp)
        hps[hp] = this_val
    return hps


def convert_statfile_to_new_format(filename):
    hps = get_hps_from_filename(filename)
    if 'cifar10' in filename:
        hps['dataset'] = 'cifar10'
    if int(hps['rals'])!=0:
        short_expname='_randomlab'
    elif int(hps['inls'])!=0:
        short_expname='_indivlab'
    else:
        short_expname='_cleanlab'
    short_expname += f'_laim{hps["laim"]}' if hps['laim'] is not None and int(hps['laim'])>0 else ''
    short_expname += f'_lang{hps["lang"]}' if hps['lang'] is not None and float(hps['lang'])>0 else ''
    short_expname += f'_lall{hps["lall"]}' if hps['lall'] is not None and float(hps['lall'])>0 else ''
    short_expname += f'_smze{hps["smze"]}' if hps['smze'] is not None and int(hps['smze'])>0 else ''
    short_expname += f'_with{hps["with"]}'
    short_expname += f'_lr{hps["lr"]}'

    temp_stats = myload(filename)
    if isinstance(temp_stats, list):
        tag_dict, temp_stats = temp_stats

    mysave('./stats/'+'cifar10/',f'config_mlp_randomlabels_raep{hps["nend"]}_llep{hps["nell"]}_seed{hps["seed"]}_cifar10'+short_expname+'_0.txt', hps)
    mysave('./stats/'+'cifar10/',f'stats_mlp_randomlabels_raep{hps["nend"]}_llep{hps["nell"]}_seed{hps["seed"]}_cifar10'+short_expname+'_0.txt', temp_stats)



def find(pattern, path, nosub = False):
    '''
    Returns list of filenames containing pattern in path.
    '''
    result = []
    if nosub:
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for name in all_files:
            if fnmatch.fnmatch(name,pattern):
                result.append(os.path.join(path, name))
    else:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
    return result


def mysave(path,name,data):
    if os.path.exists(path):
        if os.path.exists(path+name):
            os.remove(path+name)
        with open(path+name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)
    else:
        os.makedirs(path)
        with open(path + name, "wb") as fp:   #Pickling
            pickle.dump(data, fp)

def myload(name, device = None):
    with open(name, "rb") as fp:   # Unpickling
        if device is None:
            all_stats = pickle.load(fp)
        else:
            all_stats = pickle.load(name,map_location=device)
    return all_stats

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
    
def compute_accuracy(model, data_loader, device=None):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            if len(targets.size())>1:
                targets = torch.where(targets==1)[1]#.type(predicted_labels.dtype)
            if device is not None:
                features = features.to(device)
                targets = targets.to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            # correct_pred += (predicted_labels.cpu() == targets.cpu()).sum()
    return correct_pred.float() / num_examples * 100

def load_multiseed_stats(filenames):
    all_stats,all_configs={},{}
    if os.path.exists(filenames[0].replace('stats_','config_')):
        for filename in filenames:
            # ll_filename = find(f'mlp*paam='+param+'*perb='+perturb+'*opim='+optimalgo+f'*seed={seed}-*','stats/cifar10/')[0]#'stats/scaletracking/'
            #n_epochs = np.int64(filename.split('-nehs=',2)[1].split('-',2)[0])
            try:
                temp_stats = myload(filename)
                if isinstance(temp_stats, list):
                    tag_dict, temp_stats = temp_stats
            except RuntimeError: continue
            # if all_stats == {}:
            #     all_stats = temp_stats
            # else:
            for key in temp_stats.keys():
                if len(temp_stats[key]['epoch']) < 21: continue
                if key not in all_stats.keys():
                    all_stats[key] = {}
                    hps = myload(filename.replace('stats_', 'config_'))
                    all_configs[key] = hps
                
                for subkey in temp_stats[key].keys():
                    if subkey in all_stats[key].keys():
                        all_stats[key][subkey].append(temp_stats[key][subkey])
                    else:
                        all_stats[key][subkey]=[temp_stats[key][subkey]]
        # return the same shape as if it had multiple seeds:
        for key in all_stats.keys():
            try:
                if all_stats != {} and len(np.array(all_stats[key]['epoch']).shape)==1:
                    for thiskey in all_stats:
                        for subkey in all_stats[thiskey]:
                            all_stats[thiskey][subkey] = [all_stats[thiskey][subkey]]
            except ValueError:
                num_eps = [len(thisstat) for thisstat in all_stats[key]['epoch']]
                raise ValueError(f'Epochs in different runs dont coincide for {key}: {num_eps}\n {all_configs[key]}')

        try:
            return all_stats, hps
        except UnboundLocalError:
            return None, None
    elif '-nehs=' in filenames[0]:
        for filename in filenames:
            # ll_filename = find(f'mlp*paam='+param+'*perb='+perturb+'*opim='+optimalgo+f'*seed={seed}-*','stats/cifar10/')[0]#'stats/scaletracking/'
            n_epochs = np.int64(filename.split('-nehs=',2)[1].split('-',2)[0])
            try:
                temp_stats = myload(filename)
                if isinstance(temp_stats, list):
                    tag_dict, temp_stats = temp_stats
            except RuntimeError: continue
            # if all_stats == {}:
            #     all_stats = temp_stats
            # else:
            for key in temp_stats.keys():
                if key not in all_stats.keys():
                    all_stats[key] = {}
                
                for subkey in temp_stats[key].keys():
                    if subkey in all_stats[key].keys():
                        all_stats[key][subkey].append(temp_stats[key][subkey])
                    else:
                        all_stats[key][subkey]=[temp_stats[key][subkey]]
        # return the same shape as if it had multiple seeds:
        for key in all_stats.keys():
            if all_stats != {} and len(np.array(all_stats[key]['epoch']).shape)==1:
                for thiskey in all_stats:
                    for subkey in all_stats[thiskey]:
                        all_stats[thiskey][subkey] = [all_stats[thiskey][subkey]]
        try:
            return all_stats, n_epochs
        except UnboundLocalError:
            return None, None
    else:
        print('Unknown saving format.')

def load_multiseed_stats_oldold(filenames):
    all_stats={}
    for filename in filenames:
        try:
            temp_stats = myload(filename)
            if isinstance(temp_stats, list):
                tag_dict, temp_stats = temp_stats
        except RuntimeError: continue
        if all_stats == {}:
            all_stats = temp_stats
        else:
            for key in temp_stats.keys():
                if key not in all_stats.keys():
                    all_stats[key] = temp_stats[key]
                else:
                    for subkey in temp_stats[key].keys():
                        if subkey in all_stats[key].keys():
                            all_stats[key][subkey].extend(temp_stats[key][subkey])
                        else:
                            all_stats[key][subkey]=temp_stats[key][subkey]
    # if not multiple seeds, make sure still same shape
    for key in all_stats.keys():
        if all_stats != {} and len(np.array(all_stats[key]['epoch']).shape)==1:
            for thiskey in all_stats:
                for subkey in all_stats[thiskey]:
                    all_stats[thiskey][subkey] = [all_stats[thiskey][subkey]]
    return all_stats
    

def process_stats(stats, epoch_count = False, picklast=False):
    mlp_iters = {}
    mlp_stats = {}
    for key in stats:
        mlp_iters[key] = {}
        mlp_stats[key] = {'epoch': stats[key]['epoch'], 'iter': stats[key]['iter']}
        for subkey in stats[key]:
            if subkey in ['epoch','iter']: continue
            #print(subkey)
            if picklast:
                try:
                # print(subkey, stats[key][subkey][0])
                    mlp_iters[key][subkey] = [stats[key][subkey][irun][-1][0] for irun in range(len(stats[key][subkey]))]
                    mlp_stats[key][subkey] = [stats[key][subkey][irun][-1][1] for irun in range(len(stats[key][subkey]))]
                except Exception as ex:
                    mlp_iters[key][subkey] = [np.nan for irun in range(len(stats[key][subkey]))]
                    mlp_stats[key][subkey] = [np.nan for irun in range(len(stats[key][subkey]))]
                    print(key, subkey, ' caught exception ',ex)
                continue
            try:
                mlp_iters[key][subkey] = [stat[0] for stat in stats[key][subkey][0] if stat[2]==epoch_count]
                mlp_stats[key][subkey] = [[stat[1] for stat in stats[key][subkey][irun] if stat[2]==epoch_count] for irun in range(len(stats[key][subkey]))]
            except TypeError:
                mlp_iters[key][subkey] = [np.nan]
                mlp_stats[key][subkey] = [[np.nan] for irun in range(len(stats[key][subkey]))]
    return mlp_iters, mlp_stats



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
#contents = CPU_Unpickler(f).load()

def loads(x):
    bs = io.BytesIO(x)
    unpickler = CPU_Unpickler(bs)
    return unpickler.load()

def fix(map_loc):
    # Closure rather than a lambda to preserve map_loc 
    return lambda b: torch.load(io.BytesIO(b), map_location=map_loc)

class MappedUnpickler(pickle.Unpickler):
    # https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219

    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return fix(self._map_location)
        else: 
            return super().find_class(module, name)

def mapped_loads(s, map_location='cpu'):
    bs = io.BytesIO(s)
    unpickler = MappedUnpickler(bs, map_location=map_location)
    return unpickler.load()



def get_date_from_seed(seed):
    #invert: seed = int(time.time()/1000)
    return time.strftime('%Y-%m-%d', time.localtime(1000 * seed))
