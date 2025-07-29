# %%
"""
Plot width versus the optimal loss for sp and mup, mse loss and ce loss.
Requires precomputed stats of full and fair learning rate sweeps for each width, loss and parameterization.
"""

import seaborn as sns 
import os
import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import find, load_multiseed_stats, myload
from utils.plot_utils import adjust_lightness, adjust_fontsize
from utils.eval import exponent, scaling_law
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
from matplotlib.lines import Line2D

brightnesses= np.linspace(0.5,1.75,4)
plot_path = 'figures/2dplots/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
colors = ['tab:blue','tab:orange']
markers = ['x', 'o', '+', 'D', '*', 'p', 'H', 's']
no_zoom = True #False
fill = False #True

adjust_fontsize(3)

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



# C10
# sgd CELoss mup (191), sp (390)
# sgd MSELoss mup (1000), sp (1010)
# adam CELoss mup (2293), sp (271)
# adam MSELoss mup (1100), sp (1110,295)

# MNIST
# sgd CELoss mup (4000), sp (990)
# sgd MSELoss mup (2000), sp (990)
# adam CELoss mup (4100), sp (971)
# adam MSELoss mup (2100), sp (900)

epoch = 1
thisstat = 'Loss/train'
all_best_losses = {}
for seed in [
    191, 390, 1000, 1100, 1010,2293,271,1110,
    4000, 4100,2000,2100,'990_mse', '990_ce',971,900
    ]:

    if seed in [191,1000,2293,1100,4000,2000,4100,2100]:
        param = 'mup'
    else:
        param = 'sp'
    if seed in [191,390,2293,271,4000,4100,971]:
        loss='ce'
    else:
        loss = 'mse'
    if seed in [191,390,1000,1010,4000,2000,990]:
        optim = 'SGD'
    else:
        optim = 'ADAM'
    if seed in [191,390,1000,1010,1100,1110,2293,271]:
        dataset = 'cifar10'
    else:
        dataset = 'mnist'
    
    if seed == '990_mse':
        seed = 990
        optim = 'SGD'
        param = 'sp'
        dataset = 'mnist'
        loss = 'mse'
    elif seed == '990_ce':
        seed = 990
        optim = 'SGD'
        param = 'sp'
        dataset = 'mnist'
        loss = 'ce'


    filter_string = f'stats_mlp*seed{seed}_*'

    filenames=[]
    for filenam in find(filter_string, './stats/'+dataset+'/'):
        #config = myload(filenam.replace('stats_','config_'))
        checkdataset = dataset+'_mse' if loss == 'mse' else dataset
        if check_compatibility(filenam.replace('stats_','config_'), dataset=checkdataset, n_hiddenlayers = 7, optim_algo = optim,param=param):
            filenames.append(filenam)
    if len(filenames) == 0:
        print(f'No files found for {seed}, {dataset}, {optim}, {param}')
        continue

    mlp_stats, hps = load_multiseed_stats(filenames)
    if seed == 990:
        loss = 'mse' if 'mse' in hps.dataset else 'ce'

    if mlp_stats is None or len(mlp_stats.keys()) == 0:
        print(f'No stats found for {seed}, {dataset}, {optim}, {param}')
        continue


    WIDTHS, LRS, RHOS=[],[],[]
    for key in sorted(mlp_stats, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS: WIDTHS.append(key[0])
        if key[1] not in LRS: LRS.append(key[1])
        if key[2] not in RHOS: RHOS.append(key[2])
            
    WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
    #print(WIDTHS,LRS,RHOS)
    for WIDTH in WIDTHS:
        final_stats = {}
        for key in mlp_stats:
            if key[0] != WIDTH: continue
            if len(mlp_stats[key][thisstat][0][0])==3:
                for irun in range(len(mlp_stats[key][thisstat])):
                    iters, accs = read_stats(mlp_stats[key][thisstat][irun], epoch_count=True)
                    if key[1] in final_stats:
                        final_stats[key[1]].append(accs[epoch])
                    else:
                        final_stats[key[1]] = [accs[epoch]]
            #final_stats[(key[0],key[1])] = np.array(final_stats[key[1]])
            all_best_losses[(loss,optim,param,seed,WIDTH,key[1],dataset)] = np.mean(final_stats[key[1]])
            # print(final_stats[(key[0],key[1])].shape) # nseeds x nepochs

print([key for key in all_best_losses if key[2] == 'mup' and key[4] == 256])
# mse vs ce for best LR at each (width, param, dataset)
for optim in ['SGD','ADAM']:
    for dataset in ['cifar10','mnist']:    
        for param in ['mup','sp']:
            for width in WIDTHS:
                print(optim,param,dataset,width)
                try:
                    maxacc_ce = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'ce' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                    maxacc_mse = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'mse' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                    print('ce: ', maxacc_ce)
                    print('ce-mse: ', maxacc_ce-maxacc_mse)
                except ValueError:
                    print('missing stats')


#%%
# plot ce-mse across widths for mup vs sp (color) and optim (linestyle) (1 subplot mnist and one c10)
# for each (dataset,optim): plot mup vs sp (color) and loss (linestyle) across widths

import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import get_filename_from_args, mysave, myload, find
from utils.plot_utils import adjust_lightness, width_plot, adjust_fontsize
from utils.eval import exponent, scaling_law
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
brightnesses= np.linspace(0.5,1.75,4)
plot_path = './figures/coord_checks/' 
if  not os.path.exists(plot_path):
    os.makedirs(plot_path)
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"mathtext.fontset": 'cm'})

colors = ['tab:blue','tab:orange']
linestyles = ['-','--']
adjust_fontsize(3.5)

WIDTHS = [256,1024,4096]#,16384]

optim='SGD'

# joint plot
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(2*onefigsize[0],1*onefigsize[1]))
brightnesses = np.linspace(0.5,1.75,len(WIDTHS))

for iplt, (dataset,optim) in enumerate([('mnist','SGD'),('cifar10','SGD')]):
    for iparam, param in enumerate(['mup','sp']):
        max_ces, max_mses = [], []
        for width in WIDTHS:
            #print(optim,param,dataset,width)
            try:
                maxacc_ce = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'ce' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                maxacc_mse = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'mse' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                max_ces.append(maxacc_ce)
                max_mses.append(maxacc_mse)
            except ValueError:
                print('missing stats')
                max_ces.append(np.nan)
                max_mses.append(np.nan)
        try:
            ax[iplt].plot(WIDTHS, max_ces, label=f'{param}, ce',color = colors[iparam])
            ax[iplt].plot(WIDTHS, max_mses,'--',label=f'{param}, mse',color = colors[iparam])
        except ValueError:
            print('missing stats')
        ax[iplt].set_xscale('log')
        ax[iplt].set_xticks(WIDTHS)
        ax[iplt].set_xticklabels(WIDTHS)
        ax[iplt].set_xlabel(f'Width')

ax[0].set_title('MNIST')
ax[1].set_title('CIFAR-10')
#ymax = np.nanmax(train_losses)

thisylabel = 'Train accuracy'
ax[0].set_ylabel(thisylabel)

adjust_fontsize(2.8)
ax[0].legend(title='param, loss',frameon=True, loc='center')

plt.savefig('./figures/'+f'losses_across_widths_mupvssp_joint_{optim}_largefont.png', dpi=300, bbox_inches='tight')

# %%
for iplt, (dataset,optim) in enumerate([('mnist','SGD'),('cifar10','SGD'),('mnist','ADAM'),('cifar10','ADAM')]):
    fig,ax=plt.subplots(1,1)
    brightnesses = np.linspace(0.5,1.75,len(WIDTHS))

    for iparam, param in enumerate(['mup','sp']):
        max_ces, max_mses = [], []
        for width in WIDTHS:
            #print(optim,param,dataset,width)
            try:
                maxacc_ce = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'ce' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                maxacc_mse = np.max([all_best_losses[key] for key in all_best_losses if key[0] == 'mse' and key[1] == optim and key[2] == param and key[4] == width and dataset in key[6]])
                max_ces.append(maxacc_ce)
                max_mses.append(maxacc_mse)
            except ValueError:
                print('missing stats')
                max_ces.append(np.nan)
                max_mses.append(np.nan)
        try:
            ax.plot(WIDTHS, max_ces, label=f'{param}, ce',color = colors[iparam])
            ax.plot(WIDTHS, max_mses,'--',label=f'{param}, mse',color = colors[iparam])
        except ValueError:
            print('missing stats')
    plt.legend(title='param, loss')
    #ymax = np.nanmax(train_losses)

    ax.set_xscale('log')
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels(WIDTHS)
    ax.set_xlabel(f'Width')
    thisylabel = 'Training accuracy'
    ax.set_ylabel(thisylabel)
    plt.savefig('./figures/'+f'losses_across_widths_mupvssp_{dataset}_{optim}.png', dpi=300, bbox_inches='tight')

