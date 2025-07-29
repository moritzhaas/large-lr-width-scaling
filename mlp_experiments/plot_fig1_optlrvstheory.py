# %%
"""
Plot width versus the optimal learning rate for MLPs and GPT, and fit scaling laws (Figure 1).
Requires pre-computed statistics from `plot_litgpt_loss_wandb.py` loaded from wandb and MLP learning rate sweeps run with `main_mlp_allwidths.py`.

"""

# plot width vs optLR for GPT SGD and MLP
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import find, load_multiseed_stats, myload
from utils.plot_utils import adjust_fontsize
#from utils.ssh_utils import get_recent_files_via_ssh, read_hdf5_keys_via_ssh, read_hdf5_entry_via_ssh, get_stats_from_h5py_via_ssh_old, get_recent_folders_via_ssh, establish_ssh_connection, execute_with_retries
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update({'text.usetex': False})
onefigsize = bundles.icml2022()['figure.figsize']
import numpy as np
import os
from utils.eval import exponent, scaling_law
from scipy import stats


def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    return (np.log(y[0])-np.log(y[-1]))/(np.log(x[0])-np.log(x[-1]))

WIDTHS = [256, 1024, 4096, 16384]

# WARNING: run plot_litgpt_loss_wandb.py first to save these loss statistics
results_train,results_val = myload('stats/litgpt/'+f'final_losses_pythia_standard_SGD_width_lr.txt')

WIDTHS, LRS, WARMUPS=[],[],[]
for key in sorted(results_val, key=lambda key: int(key[0])):
    if key[0] not in WIDTHS: WIDTHS.append(key[0])
    if key[1] not in LRS: LRS.append(key[1])
    if key[2] not in WARMUPS: WARMUPS.append(key[2])
        
WIDTHS, LRS, WARMUPS = np.sort(WIDTHS),np.sort(LRS),np.sort(WARMUPS)
print(WIDTHS,LRS,WARMUPS)

for iw, width in enumerate(WIDTHS):
    print('Width ', width)
    for lr in LRS:
        for key in results_train.keys():
            if key[0] == width and key[1] == lr:
                print(f'lr {key[1]}: {np.mean(results_train[key])}')
            
lrs_gpt_nogradnorm = [0.31622776601683794,0.15811388300841897, 0.07905694150420949]
exp_gpt_nogradnorm = -0.5

lrs_gpt=[5.17947467923121,3.5984283650057587, 1.7992141825028793]
exp_gpt = -0.38 #-0.38135971089688003
# x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
# slope

# %%
results_train2,results_val2 = myload('stats/litgpt/'+f'final_losses_pythia_standard_qknorm_width_lr.txt')
WIDTHS, LRS, WARMUPS=[],[],[]
for key in sorted(results_train2, key=lambda key: int(key[0])):
    if key[0] not in WIDTHS: WIDTHS.append(key[0])
    if key[1] not in LRS: LRS.append(key[1])
    if key[2] not in WARMUPS: WARMUPS.append(key[2])
        
WIDTHS, LRS, WARMUPS = np.sort(WIDTHS),np.sort(LRS),np.sort(WARMUPS)
print(WIDTHS,LRS,WARMUPS)

for iw, width in enumerate(WIDTHS):
    print('Width ', width)
    for lr in LRS:
        for key in results_train2.keys():
            if key[0] == width and key[1] == lr:
                print(f'lr {key[1]}: {np.mean(results_train2[key])}')

lrs_gpt_adam = [0.01, 0.007905694150420948, 0.00316227766]
exp_gpt_adam = -0.42 # exponent(WIDTHS[:3],lrs_gpt_adam) #-0.41524101188012513
# x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
# slope == -0.4152410118801249

# %%

def read_stats(thisstat,epoch_count = False):
        thisiters = [stat[0] for stat in thisstat if stat[2]==epoch_count]
        thisval = [stat[1] for stat in thisstat if stat[2]==epoch_count]
        return thisiters, thisval

def check_compatibility(configname, SEED=None, dataset=None, activ=None, label_noise = None, n_hiddenlayers = 2, optim_algo='SGD', use_bias=False, residual=False, param='sp'):
    hps = myload(configname)
    if SEED is not None and int(hps.seed) != SEED: return False
    if label_noise is not None and label_noise != hps.label_noise: return False
    if dataset is not None and dataset != hps.dataset: return False
    partial_check = (hps.optim == optim_algo and 
        int(hps.nhiddenlayers) == n_hiddenlayers and 
        hps.param == param and 
        hps.optim == optim_algo and 
        hps.use_bias == use_bias)
    #print(configname, partial_check)
    try:
        if activ is not None and 'lin' in activ:
            activ = hps.linear
        elif activ is None and not hps.linear:
            activ = True
        else:
            activ = False
    except AttributeError:
        activ = activ is None or not ('lin' in activ)
    try:
        res_match = (residual == hps.resnet)
    except AttributeError:
        res_match = (not residual)
    return partial_check and activ and res_match

def load_multiseed_stats(filenames,to_epoch=1):
    all_stats,all_configs={},{}
    if os.path.exists(filenames[0].replace('stats_','config_')):
        for filename in filenames:

            try:
                temp_stats = myload(filename)
                if isinstance(temp_stats, list):
                    tag_dict, temp_stats = temp_stats
            except RuntimeError: continue

            for key in temp_stats.keys():
                #if 'cifar10' in filename and len(temp_stats[key]['epoch']) < 21 and SEEDS[0]==90: continue
                if key not in all_stats.keys():
                    all_stats[key] = {}
                    hps = myload(filename.replace('stats_', 'config_'))
                    all_configs[key] = hps
                
                for subkey in temp_stats[key].keys():
                    if subkey in all_stats[key].keys():
                        all_stats[key][subkey].append(temp_stats[key][subkey][:to_epoch+1])
                    else:
                        all_stats[key][subkey]=[temp_stats[key][subkey][:to_epoch+1]]
        # return the same shape as if it had multiple seeds:
        for key in all_stats.keys():
            try:
                if all_stats != {} and len(np.array(all_stats[key]['epoch']).shape)==1:
                    for thiskey in all_stats:
                        for subkey in all_stats[thiskey]:
                            all_stats[thiskey][subkey] = [all_stats[thiskey][subkey][:to_epoch+1]]
            except ValueError:
                num_eps = [len(thisstat) for thisstat in all_stats[key]['epoch']]
                raise ValueError(f'Epochs in different runs dont coincide for {key}: {num_eps}\n {all_configs[key]}')
        try:
            return all_stats, hps
        except UnboundLocalError:
            return None, None
    elif '-nehs=' in filenames[0]:
        for filename in filenames:
            n_epochs = np.int64(filename.split('-nehs=',2)[1].split('-',2)[0])
            try:
                temp_stats = myload(filename)
                if isinstance(temp_stats, list):
                    tag_dict, temp_stats = temp_stats
            except RuntimeError: continue
            for key in temp_stats.keys():
                if key not in all_stats.keys():
                    all_stats[key] = {}
                
                for subkey in temp_stats[key].keys():
                    if subkey in all_stats[key].keys():
                        all_stats[key][subkey].append(temp_stats[key][subkey][:to_epoch+1])
                    else:
                        all_stats[key][subkey]=[temp_stats[key][subkey][:to_epoch+1]]
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

def get_best_lrs_from_stats(SEEDS, dataset, n_hidden=7,optim_algo='SGD',stat_path=None,min_acc=20):
    filter_strings = [f'stats_mlp*seed{SEED}_*' for SEED in SEEDS]
    if stat_path is None:
        stat_path = './stats/'+dataset+'/'

    filenames=[]
    for filter_string in filter_strings:
        for filenam in find(filter_string, stat_path):
            if check_compatibility(filenam.replace('stats_','config_'), n_hiddenlayers = n_hidden, activ= None, optim_algo = optim_algo,param='sp',dataset=dataset):
                filenames.append(filenam)

    if len(filenames) == 0:
        print(f'No files found '+filter_strings[0])

    mlp_stats, hps = load_multiseed_stats(filenames)

    print(SEEDS, dataset, n_hidden,optim_algo,stat_path)

    if mlp_stats is None or len(mlp_stats.keys()) == 0:
        print(f'No stats found')

    WIDTHS, LRS, RHOS=[],[],[]
    for key in sorted(mlp_stats, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS: WIDTHS.append(key[0])
        if key[1] not in LRS: LRS.append(key[1])
        if key[2] not in RHOS: RHOS.append(key[2])
            
    WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
    #print(WIDTHS,LRS,RHOS)

    ll_stats = {}
    thisstat = 'Loss/train'
    try:
        for key in mlp_stats:
            if len(mlp_stats[key][thisstat][0][0])==3:
                for irun in range(len(mlp_stats[key][thisstat])):
                    iters, accs = read_stats(mlp_stats[key][thisstat][irun], epoch_count=True)
                    if (key[0],key[1]) in ll_stats:
                        ll_stats[(key[0],key[1])].append(accs)
                    else:
                        ll_stats[(key[0],key[1])] = [accs]
            ll_stats[(key[0],key[1])] = np.array(ll_stats[(key[0],key[1])])
            # print(ll_stats[(key[0],key[1])].shape) # nseeds x nepochs
    except:
        for key in sorted(mlp_stats, key=lambda key: int(-key[0])): break

    opt_lrs, opt_accs,max_lrs = {}, {}, {}
    for iw, width in enumerate(WIDTHS):
        print('Width ', width)
        OPTLR,OPTACC,MAXLR=0,0,1e8
        for lr in LRS:
            for key in ll_stats.keys():
                if key[0] == width and key[1] == lr:
                    thisacc = np.mean(ll_stats[key])
                    if thisacc>OPTACC:
                        OPTLR=key[1]
                        OPTACC=thisacc
                    print(f'lr {key[1]}: {thisacc}')

        for lr in LRS:
            for key in ll_stats.keys():
                if key[0] == width and key[1] == lr:
                    thisacc = np.mean(ll_stats[key])
                    if thisacc<min_acc and key[1]< MAXLR and key[1]>=OPTLR:
                        MAXLR=key[1]
        
        opt_lrs[width] = OPTLR
        opt_accs[width] = OPTACC
        max_lrs[width] = MAXLR if MAXLR<1e6 else np.nan
    
    print(opt_lrs)
    return list(opt_lrs.keys()), list(opt_lrs.values()), list(max_lrs.values())

lrs_mnist = [0.128, 0.04525483, 0.04525484, 0.064, 0.032, 0.02262742, 0.01131371]
widths_mnist = [256, 512, 1024, 2048, 4096, 8192, 16384]

# C10 8layer MLPs
SEEDS = np.arange(5321,5325)
widths_c10,lrs_c10, max_lrs_c10 = get_best_lrs_from_stats(SEEDS,'cifar10')

#%%
adjust_fontsize(3)

plot_path = './figures/mainfigures/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red']#['#1f77b4','#ff7f0e']
WIDTHS = [256, 1024, 4096, 16384]

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
iline=0

nocifar = True

x_log,y_log = np.log10(widths_mnist), np.log10(lrs_mnist)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(widths_mnist, y_log_pred,label='MLP SGD' if nocifar else 'MLP SGD MNIST',color=colors[iline])
axes.scatter(widths_mnist,lrs_mnist,s=12,marker='x', color=colors[iline],)
axes.text(widths_mnist[2],lrs_mnist[2]*1.5,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
axes.plot(widths_mnist,[10**(slope * (x_log[0]) + intercept)*(256/width) for width in widths_mnist],'--',color='tab:gray')
axes.text(widths_mnist[-1],1.2*10**(slope * (x_log[0]) + intercept)*(256/widths_mnist[-1]),f'{-1}',color='tab:gray', ha='center',fontsize=12, fontweight='bold')
iline+=1

if not nocifar:
    x_log,y_log = np.log10(widths_c10), np.log10(lrs_c10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    y_log_pred = 10**(slope * (x_log) + intercept)
    axes.plot(widths_c10, y_log_pred,label='MLP SGD CIFAR10',color=colors[iline])
    axes.scatter(widths_c10,lrs_c10,s=12,marker='x', color=colors[iline],)
    axes.text(widths_c10[2],lrs_c10[2]*0.3,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
    axes.plot(widths_c10,[10**(slope * (x_log[0]) + intercept)*(256/width) for width in widths_c10],'--',color='tab:gray')
    axes.text(widths_c10[-1],1.2*10**(slope * (x_log[0]) + intercept)*(256/widths_c10[-1]),f'{-1}',color='tab:gray', ha='center',fontsize=12, fontweight='bold')
    iline+=1


WIDTHS = [256, 1024, 4096,]
x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,label='GPT SGD',color=colors[iline])
axes.scatter(WIDTHS,lrs_gpt,s=12,marker='x', color=colors[iline],)
axes.text(WIDTHS[1],lrs_gpt[1]*1.2,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
axes.plot(WIDTHS,[10**(slope * (x_log[0]) + intercept)*(256/width) for width in WIDTHS],'--',color='tab:gray')
axes.text(WIDTHS[1],0.4*10**(slope * (x_log[0]) + intercept)*(256/WIDTHS[1]),f'{-1}',color='tab:gray', ha='center',fontsize=12, fontweight='bold')
iline+=1

WIDTHS = [256, 1024, 4096,]
x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,label='GPT ADAM',color=colors[iline])
axes.scatter(WIDTHS,lrs_gpt_adam,s=12,marker='x', color=colors[iline],)
axes.text(WIDTHS[1],lrs_gpt_adam[1]*1.25,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
axes.plot(WIDTHS,[10**(slope * (x_log[0]) + intercept)*(256/width) for width in WIDTHS],'--',color='tab:gray')
axes.text(WIDTHS[-1],1.2*10**(slope * (x_log[0]) + intercept)*(256/WIDTHS[-1]),f'{-1}',color='tab:gray', ha='center',fontsize=12, fontweight='bold')
iline+=1

#axes.set_xticks([], minor=True)
axes.set_xscale('log')
axes.set_yscale('log')
WIDTHS = [256,1024,4096,16384]
axes.set_xticks(WIDTHS)
axes.set_xticklabels(WIDTHS)
axes.set_xlabel('Width')
axes.set_ylabel('Optimal learning rate')
#axes.set_title('MLPs SGD')
#axes.set_ylim(0.0001,50)
plt.legend(frameon=True)#loc='lower left')
plt.savefig(plot_path + f'mainfig1_lrexponents_mlp_gpt{"_nocifar" if nocifar else ""}_largefont.png',dpi=300)


#%%