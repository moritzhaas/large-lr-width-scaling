# %%

"""
Plot the statistics tracked with extended eval over the course of training for each width (the darker, the wider)
"""

import numpy as np 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import find, load_multiseed_stats
from utils.plot_utils import adjust_lightness
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']

brightnesses= np.linspace(0.5,1.75,4)
plot_path = './figures/across_training/' 
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"mathtext.fontset": 'cm'})

#MLPs
SEEDS, OPTIM_ALGO, PARAMETERIZATION, PERTURBATION = [152,153,154,155],'SGD','sp','naive'
SEEDS2, OPTIM_ALGO2, PARAMETERIZATION2, PERTURBATION2 = [152,153,154,155],'SGD','mup','mpp'
SEEDS3, OPTIM_ALGO3, PARAMETERIZATION3, PERTURBATION3 = [],'SGD','sp','naive'
resnet=False



extended_eval_iter = 20
toiter = None  #20 #None
logscale = toiter is None
fromiter = None
ylogscale = True
fillbetween = True
coarsen = resnet
color1 = 'tab:orange'
color2 = 'tab:blue'

compare = (SEEDS2 != [])
third_baseline = (SEEDS3 != [])

if resnet:
    num_plots = 4
else:
    num_plots = 3

filenames, filenames2, filenames3 = [], [], []
for SEED in SEEDS:
    for filenam in find(f'mlp-*paam={PARAMETERIZATION}-*perb={PERTURBATION}-*opim={OPTIM_ALGO}-*seed={SEED}-{"*exer="+str(extended_eval_iter) if extended_eval_iter is not None else ""}*', './stats/cifar10/'):
        filenames.append(filenam)

for SEED2 in SEEDS2:
    for filenam in find(f'mlp-*paam={PARAMETERIZATION2}-*perb={PERTURBATION2}-*opim={OPTIM_ALGO2}-*seed={SEED2}-{"*exer="+str(extended_eval_iter) if extended_eval_iter is not None else ""}*', './stats/cifar10/'):
        filenames2.append(filenam)

for SEED3 in SEEDS3:
    for filenam in find(f'mlp-*paam={PARAMETERIZATION3}-*perb={PERTURBATION3}-*opim={OPTIM_ALGO3}-*seed={SEED3}-{"*exer="+str(extended_eval_iter) if extended_eval_iter is not None else ""}*', './stats/cifar10/'):
        filenames3.append(filenam)

mlp_stats, N_EPOCHS = load_multiseed_stats(filenames)
if compare:
    mlp_stats2, N_EPOCHS2 = load_multiseed_stats(filenames2)
if third_baseline:
    sgd_stats, N_EPOCHS_SGD = load_multiseed_stats(filenames3)

# find all widths, lrs, rhos from stat keys:
WIDTHS, LRS, RHOS=[],[],[]
for key in sorted(mlp_stats, key=lambda key: int(key[0])):
    if key[0] not in WIDTHS: WIDTHS.append(key[0])
    if key[1] not in LRS: LRS.append(key[1])
    if key[2] not in RHOS: RHOS.append(key[2])
        
WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
print(WIDTHS,LRS,RHOS)

if len(WIDTHS)>=6:
    WIDTHS= WIDTHS[:5]
    for key in sorted(mlp_stats, key=lambda key: int(key[0])):
        if key[0]>8192: del mlp_stats[key]

for key in sorted(mlp_stats, key=lambda key: int(-key[0])): break
if compare:
    for key2 in sorted(mlp_stats2, key=lambda key: int(-key[0])): break
    N_RUNS2 = len(mlp_stats2[key2][f'Loss/train'])
if third_baseline:
    for sgdkey in sorted(sgd_stats, key=lambda key: int(-key[0])): break

def read_stats(thisstat,epoch_count = False):
    thisiters = [stat[0] for stat in thisstat if stat[2]==epoch_count]
    thisval = [stat[1] for stat in thisstat if stat[2]==epoch_count]
    return thisiters, thisval

# gradnorms = np.mean(mlp_stats[key]['gradnorms'],axis=0).shape # iter x layer
iters, val_acc = read_stats(mlp_stats[key]['Loss/val'][0])
iters2, val_acc2 = read_stats(mlp_stats2[key2]['Loss/val'][0])



def process_stats(stats, epoch_count = False):
    '''
    Splits up iters and stats to make sure correct iter is matched to each stat
    
    Returns format:
    stats[(256, 1.0, 0.0)]['Loss/val'] contains list of shape (irun x iter)
    '''
    mlp_iters = {}
    for key in stats:
        mlp_iters[key]={}
        for subkey in stats[key]:
            if subkey in ['epoch','iter']: continue
            mlp_iters[key][subkey] = [stat[0] for stat in stats[key][subkey][0] if stat[2]==epoch_count]
            for irun in range(len(stats[key][subkey])):
                stats[key][subkey][irun] = [stat[1] for stat in stats[key][subkey][irun] if stat[2]==epoch_count]
    return mlp_iters, stats

mlp_stats_raw = mlp_stats
mlp_iters, mlp_stats = process_stats(mlp_stats)

mlp_stats_raw2 = mlp_stats2
mlp_iters2, mlp_stats2 = process_stats(mlp_stats2)

# %%
N_RUNS = len(mlp_stats[key][f'Loss/train'])
N_ITER = len(iters)
N_EPOCHS = 20

#key= (WIDTHS[-1],key[1],key[2])
for subkey in mlp_stats[key].keys():
    if len(mlp_stats[key][subkey][0])>0:
        try:
            finalstats = [mlp_stats[key][subkey][irun][-1] for irun in range(N_RUNS)]
            meanfinal=np.mean(finalstats)
            print(subkey,meanfinal)
        except:
            continue

titles= [f'Layer {i_l+1}' for i_l in range(4)] if resnet else ['Input layer', 'Hidden layer','Output layer']

if toiter is not None:
    iters = iters[(0 if fromiter is None else fromiter):toiter]

for ikey, key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))): 
    if key[0] == 256:
        key256 = key

if coarsen:
    # coarsen iters to logspace:
    iters=np.array(iters)
    logspace = np.logspace(np.log10(200),np.log10(iters[-1]),275,base=10,dtype=int)

    def closest(num, arr):
        curr = arr[0]
        for index in range(len(arr)):
            if abs(num - arr[index]) < abs(num - curr):
                curr = arr[index]
        return curr

    for i_iter, thisiter in enumerate(logspace):
        logspace[i_iter] = closest(thisiter,iters)

    coarse_iters = np.concatenate((iters[:25],np.unique(logspace)))
    coarse_idcs = np.zeros_like(coarse_iters,dtype=int)
    for i_iter, thisiter in enumerate(coarse_iters):
        coarse_idcs[i_iter] = np.where(iters == thisiter)[0][0]
        
    iters = iters[coarse_idcs]
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        for subkey in mlp_stats[key].keys():
            for irun in range(len(mlp_stats[key][subkey])):
                mlp_stats[key][subkey][irun] = mlp_stats[key][subkey][irun][coarse_idcs]


def get_these_data(thesestats,theseiter,fromiter=None,toiter = None, zero_init = True):
    if toiter is None:
        thesedata = np.mean(thesestats,axis=0)
        lower = np.quantile(thesestats,0.025,axis=0)
        upper = np.quantile(thesestats,0.975,axis=0)
        if zero_init:
            thesedata = np.insert(thesedata, 0,1e-8)
            lower = np.insert(lower, 0,1e-8)
            upper = np.insert(upper, 0,1e-8)
            theseiter = np.array([thit + 1 for thit in theseiter])
            theseiter = np.insert(theseiter,0,0)
    else:
        raise NotImplementedError
    if fromiter is not None:
        raise NotImplementedError
    return thesedata, lower, upper, theseiter

# %%
# valacc
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'Loss/val'],mlp_iters[key][f'Loss/val'], fromiter, toiter,zero_init=False)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(mlp_stats2[key][f'Loss/val'],mlp_iters2[key][f'Loss/val'], fromiter, toiter,zero_init=False)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
        axes.set_xlabel('Batches of training')
if logscale: axes.set_xscale('symlog')

#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'Test accuracy')
axes.legend(title='param width',loc='lower right')
plt.savefig(plot_path+f'valacc_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# trainacc
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'Loss/train'],mlp_iters[key][f'Loss/train'], fromiter, toiter,zero_init=False)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(mlp_stats2[key][f'Loss/train'],mlp_iters2[key][f'Loss/train'], fromiter, toiter,zero_init=False)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
        axes.set_xlabel('Batches of training')
if logscale: axes.set_xscale('symlog')

#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'Train accuracy')
axes.legend(title='param width',loc='lower right')
plt.savefig(plot_path+f'trainacc_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# delta activations
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'activation_delta_layer{iplt}'],mlp_iters[key][f'activation_delta_layer{iplt}'], fromiter, toiter)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata/np.sqrt(10), label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(10),upper/np.sqrt(10),color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata/np.sqrt(key[0]), label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(key[0]),upper/np.sqrt(key[0]),color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(mlp_stats2[key][f'activation_delta_layer{iplt}'],mlp_iters2[key][f'activation_delta_layer{iplt}'], fromiter, toiter)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata/np.sqrt(10), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(10),upper/np.sqrt(10),color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata/np.sqrt(key[0]), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(key[0]),upper/np.sqrt(key[0]),color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
for iplt in range(num_plots):    
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
    if iplt<2:
        axes[iplt].set_yticks([0,0.01,0.1])
        axes[iplt].set_yticklabels([0,0.01,0.1])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'Coord.wise $\|x_t-x_0\|$')
axes[0].legend(title='width',loc='upper left')
plt.savefig(plot_path+f'activationdelta_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# plot activation sparsity
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'activ0_layer{iplt}'],mlp_iters[key][f'activ0_layer{iplt}'], fromiter, toiter,zero_init=False)
        axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}' if iplt==0 else None,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(mlp_stats2[key][f'activ0_layer{iplt}'],mlp_iters2[key][f'activ0_layer{iplt}'], fromiter, toiter,zero_init=False)
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}' if iplt==0 else None,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
axes[0].set_ylabel('Coordinates == 0')
axes[0].legend(title='param width', loc='upper left')
plt.savefig(plot_path+f'activ0_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

#%%
# np.log_fanin(||delta W x||/||delta W||_* ||x||)
# using np.log_fanin(x) = np.log(x)/np.log(fanin)
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(np.log(np.array(mlp_stats[key][f'delta_W_x_norm']) / (np.array(mlp_stats[key][f'delta_W_spectral_norms'])[:,:,-1]*np.array(mlp_stats[key]['ll_activation_norm']))) / np.log(key[0]),mlp_iters[key][f'delta_W_x_norm'], fromiter, toiter)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.log(np.array(mlp_stats2[key][f'delta_W_x_norm']) / (np.array(mlp_stats2[key][f'delta_W_spectral_norms'])[:,:,-1]*np.array(mlp_stats2[key]['ll_activation_norm'])))/ np.log(key[0]),mlp_iters2[key][f'delta_W_x_norm'], fromiter, toiter)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)    
    axes.set_xlabel('Batches of training')
    if logscale: axes.set_xscale('symlog')
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'$log_n(\|\Delta W x\|/(\|\Delta W\|_*\|x\|))$')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'deltaWx_logalignment_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()


# np.log_fanin(||delta W x||/||delta W||_F ||x||)
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(np.log(np.array(mlp_stats[key][f'delta_W_x_norm']) / (np.array(mlp_stats[key][f'delta_W_frob_norms'])[:,:,-1]*np.array(mlp_stats[key]['ll_activation_norm']))) / np.log(key[0]),mlp_iters[key][f'delta_W_x_norm'], fromiter, toiter)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.log(np.array(mlp_stats2[key][f'delta_W_x_norm']) / (np.array(mlp_stats2[key][f'delta_W_frob_norms'])[:,:,-1]*np.array(mlp_stats2[key]['ll_activation_norm'])))/ np.log(key[0]),mlp_iters2[key][f'delta_W_x_norm'], fromiter, toiter)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)    
    axes.set_xlabel('Batches of training')
    if logscale: axes.set_xscale('symlog')
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'$log_n(\|\Delta W x\|/(\|\Delta W\|_F\|x\|))$')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'deltaWx_logalignment_frob_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()


# ||\Delta W||_F / ||\Delta W||_*
not_firstupdate = True
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'delta_W_frob_norms'])[:,:,iplt]/np.array(mlp_stats[key][f'delta_W_spectral_norms'])[:,:,iplt], mlp_iters[key][f'delta_W_frob_norms'], fromiter, toiter,zero_init=False)
        if iters[0] == 0 and not_firstupdate:
            thesedata, lower, upper, iters = thesedata[1:], lower[1:], upper[1:], iters[1:]
        if iplt == 2:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'delta_W_frob_norms'])[:,:,iplt]/np.array(mlp_stats2[key][f'delta_W_spectral_norms'])[:,:,iplt], mlp_iters2[key][f'delta_W_frob_norms'], fromiter, toiter,zero_init=False)
            #print(iplt, iters[0], thesedata[0])
            if iters[0] == 0 and not_firstupdate:
                thesedata, lower, upper, iters = thesedata[1:], lower[1:], upper[1:], iters[1:]
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale:
        if not_firstupdate:
            axes[iplt].set_xscale('log')
        else:
            axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'$\|\Delta W\|_F/\|\Delta W\|_*$')
axes[0].legend(title='width',loc='upper left')
plt.savefig(plot_path+f'deltaW_Foverspectral_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()
plt.close('all')


# activation alignment, same point as input
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'activ_align_init_samept']) ,mlp_iters[key][f'activ_align_init_samept'], fromiter, toiter,zero_init=False)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'activ_align_init_samept']),mlp_iters2[key][f'activ_align_init_samept'], fromiter, toiter, zero_init=False)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)    
    axes.set_xlabel('Batches of training')
    if logscale: axes.set_xscale('symlog')
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'$x^L_0 \cdot x^L_t$, same input')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'activ_align_init_samept_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# activation alignment, other training point as input
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'activ_align_init_otherpt']),mlp_iters[key][f'activ_align_init_otherpt'], fromiter, toiter,zero_init=False)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'activ_align_init_otherpt']),mlp_iters2[key][f'activ_align_init_otherpt'], fromiter, toiter, zero_init=False)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)    
    axes.set_xlabel('Batches of training')
    if logscale: axes.set_xscale('symlog')
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'$x^L_0 \cdot x^L_t$, other input')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'activ_align_init_otherpt_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# hessiannorm
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'hessiannorm']),mlp_iters[key][f'hessiannorm'], fromiter, toiter,zero_init=False)
    axes.plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'hessiannorm']),mlp_iters2[key][f'hessiannorm'], fromiter, toiter, zero_init=False)
        axes.plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)    
    axes.set_xlabel('Batches of training')
    if logscale: axes.set_xscale('symlog')
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'$\|H\|_*$')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'hessiannorm_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()



# gradnorms
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'gradnorms'])[:,:,iplt],mlp_iters[key][f'gradnorms'], fromiter, toiter,zero_init=False)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'gradnorms'])[:,:,iplt],mlp_iters2[key][f'gradnorms'], fromiter, toiter,zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'gradnorm')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'gradnorm_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()
plt.close('all')

# eta * gradnorms, where eta width-independent!
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(key[1]*np.array(mlp_stats[key][f'gradnorms'])[:,:,iplt],mlp_iters[key][f'gradnorms'], fromiter, toiter,zero_init=False)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            factors = [key[0],1,1/key[0]] if PARAMETERIZATION2=='mup' else [1,1,1]
            thesedata, lower, upper, iters = get_these_data(factors[iplt]*key[1]*np.array(mlp_stats2[key][f'gradnorms'])[:,:,iplt], mlp_iters2[key][f'gradnorms'], fromiter, toiter,zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'$\eta_l \cdot$gradnorm')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'gradnorm_etal_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()
plt.close('all')


# activation l2
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):    
        factor = 1/np.sqrt(key[0]) if iplt < 2 else 1/np.sqrt(10)
        thesedata, lower, upper, iters = get_these_data(factor * np.array(mlp_stats[key][f'activation_l2_layer{iplt}']),mlp_iters[key][f'activation_l2_layer{iplt}'], fromiter, toiter,zero_init=False)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            factor = 1/np.sqrt(key[0]) if iplt < 2 else 1/np.sqrt(10)
            thesedata, lower, upper, iters = get_these_data(factor*np.array(mlp_stats2[key][f'activation_l2_layer{iplt}']), mlp_iters2[key][f'activation_l2_layer{iplt}'], fromiter, toiter,zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'$\|$activations$\|$')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'activ_l2_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()
plt.close('all')


# reverse engineer L'(f) = ll-grad/||x^L||
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    minleniter = np.minimum(len(mlp_iters[key][f'activation_l2_layer{1}']),len(mlp_iters[key][f'gradnorms']))
    iters = np.array(mlp_iters[key][f'activation_l2_layer{1}'][:minleniter])
    if np.any(iters != np.array(mlp_iters[key][f'gradnorms'][:minleniter])):
        raise ValueError('Check iters of activation_l2_layer1 and gradnorms by hand. They do not match.')
    llgradnorms = np.array(mlp_stats[key]['gradnorms'])[:,:minleniter,-1]
    ll_activnorm = np.array(mlp_stats[key]['activation_l2_layer1'])[:,:minleniter]
    lossderivative = llgradnorms/ll_activnorm
    lower = np.quantile(lossderivative,0.025,axis=0)
    upper = np.quantile(lossderivative,0.975,axis=0)
    axes.plot(iters, lossderivative.mean(axis=0), label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
    if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
if compare:
    for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
        minleniter = np.minimum(len(mlp_iters2[key][f'activation_l2_layer{1}']),len(mlp_iters2[key][f'gradnorms']))
        iters = np.array(mlp_iters2[key][f'activation_l2_layer{1}'][:minleniter])
        if np.any(iters != np.array(mlp_iters2[key][f'gradnorms'][:minleniter])):
            raise ValueError('Check iters2 of activation_l2_layer1 and gradnorms by hand. They do not match.')
        llgradnorms = np.array(mlp_stats2[key]['gradnorms'])[:,:minleniter,-1]
        ll_activnorm = np.array(mlp_stats2[key]['activation_l2_layer1'])[:,:minleniter]
        lossderivative = llgradnorms/ll_activnorm
        lower = np.quantile(lossderivative,0.025,axis=0)
        upper = np.quantile(lossderivative,0.975,axis=0)
        axes.plot(iters, lossderivative.mean(axis=0), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
        if fillbetween: axes.fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
axes.set_xlabel('Batches of training')
if logscale: axes.set_xscale('symlog')
if ylogscale: axes.set_yscale('log')
#axes.set_ylim(2,axes[2].get_ylim()[1])
axes.set_ylabel(r'Output-Loss derivative $\chi_t$')
plt.legend(title='width')
plt.savefig(plot_path+f'lossderivative_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# %%
# plot SP self-stabilization summary: 2x3:
# Sparsity, ||x^L||, Lossderivative
# Last-layer gradnorm, ||output||, training acc

# observation: last-layer activations are quickly sparsified and shrunk in norm, reducing the diverging initial last-layer gradient.
# Toward the end of training, the training data are interpolated so that the gradient is driven to 0 through the term del L/del f.

plt.tight_layout()
fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(3*onefigsize[0],2*onefigsize[1]))

# activation sparsity
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'activ0_layer{1}'],mlp_iters[key][f'activ0_layer{1}'], fromiter, toiter,zero_init=False)
    axes[0,0].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[0,0].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[0,0].set_ylabel(r'Activ. coord.=0')

# ||x^L||
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):    
    factor = 1/np.sqrt(key[0])
    thesedata, lower, upper, iters = get_these_data(factor * np.array(mlp_stats[key][f'activation_l2_layer{1}']),mlp_iters[key][f'activation_l2_layer{1}'], fromiter, toiter,zero_init=False)
    axes[0,1].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[0,1].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[0,1].set_ylabel(r'$\|$activations$\|$')

#lossderivative
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    minleniter = np.minimum(len(mlp_iters[key][f'activation_l2_layer{1}']),len(mlp_iters[key][f'gradnorms']))
    iters = np.array(mlp_iters[key][f'activation_l2_layer{1}'][:minleniter])
    if np.any(iters != np.array(mlp_iters[key][f'gradnorms'][:minleniter])):
        raise ValueError('Check iters of activation_l2_layer1 and gradnorms by hand. They do not match.')
    llgradnorms = np.array(mlp_stats[key]['gradnorms'])[:,:minleniter,-1]
    ll_activnorm = np.array(mlp_stats[key]['activation_l2_layer1'])[:,:minleniter]
    lossderivative = llgradnorms/ll_activnorm
    lower = np.quantile(lossderivative,0.025,axis=0)
    upper = np.quantile(lossderivative,0.975,axis=0)
    axes[0,2].plot(iters, lossderivative.mean(axis=0),color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[0,2].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[0,2].set_ylabel(r'Output-Loss derivative $\chi_t$')

# LL gradnorm        
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thisnorm = np.sqrt(key[0] * 10)
    thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'gradnorms'])[:,:,2]/thisnorm,mlp_iters[key][f'gradnorms'], fromiter, toiter,zero_init=False)
    axes[1,0].plot(iters, thesedata,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[1,0].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[1,0].set_ylabel(r'Entrywise last-layer grad. norm')

# output norm 
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):    
    factor = 1/np.sqrt(10)
    thesedata, lower, upper, iters = get_these_data(factor * np.array(mlp_stats[key][f'activation_l2_layer{2}']),mlp_iters[key][f'activation_l2_layer{2}'], fromiter, toiter,zero_init=False)
    axes[1,1].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[1,1].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[1,1].set_ylabel(r'Output norm')

# training acc        
for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
    thesedata, lower, upper, iters = get_these_data(mlp_stats[key][f'Loss/train'],mlp_iters[key][f'Loss/train'], fromiter, toiter,zero_init=False)
    axes[1,2].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]))
    if fillbetween: axes[1,2].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey+1]),alpha=0.4)
axes[1,2].set_ylabel(r'Training accuracy')

for ax in axes.flatten():
    ax.set_xlabel('Batches of training')
    ax.set_xscale('symlog')
    ax.set_yscale('log')

axes[0,0].legend(title='width')
axes[1,2].legend(title='width')
plt.savefig(plot_path+f'selfstab_sp_joint_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()

# %%

if 'SAM' in OPTIM_ALGO:
    # perturbnorms
    plt.tight_layout()
    fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
    for iplt in range(num_plots):
        for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'perturbnorms'])[:,:,iplt],mlp_iters[key][f'perturbnorms'], fromiter, toiter,zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        if compare:
            for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
                thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'perturbnorms'])[:,:,iplt],mlp_iters2[key][f'perturbnorms'], fromiter, toiter,zero_init=False)
                if iplt == 2:
                    axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                    if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
                else:
                    axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                    if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
        axes[iplt].set_xlabel('Batches of training')
        if logscale: axes[iplt].set_xscale('symlog')
        #if ylogscale: axes[iplt].set_yscale('symlog')
        axes[iplt].set_title(titles[iplt])
    #axes[2].set_ylim(2,axes[2].get_ylim()[1])
    axes[0].set_ylabel(r'perturbnorms')
    plt.legend(title='width',loc='upper right')
    plt.savefig(plot_path+f'perturbnorms_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
    plt.clf()
    plt.close('all')


# feature ranks
percentile_pcs = [0.5,0.9,0.95,0.99,0.999]
ipercentile = 2
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        theserankstats = [[subsubstat[ipercentile] for subsubstat in substat] for substat in mlp_stats[key][f'feature_ranks_layer{iplt}']]
        thesedata, lower, upper, iters = get_these_data(theserankstats,mlp_iters[key][f'feature_ranks_layer{iplt}'], fromiter, toiter, zero_init=False)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata/np.sqrt(10), label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(10),upper/np.sqrt(10),color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata/np.sqrt(key[0]), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(key[0]),upper/np.sqrt(key[0]),color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            theserankstats = [[subsubstat[ipercentile] for subsubstat in substat] for substat in mlp_stats2[key][f'feature_ranks_layer{iplt}']]
            thesedata, lower, upper, iters = get_these_data(theserankstats,mlp_iters2[key][f'feature_ranks_layer{iplt}'], fromiter, toiter, zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata/np.sqrt(10), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(10),upper/np.sqrt(10),color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata/np.sqrt(key[0]), label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower/np.sqrt(key[0]),upper/np.sqrt(key[0]),color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
for iplt in range(num_plots):    
    axes[iplt].set_xlabel('Epoch')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
    #if iplt<2:
    #    axes[iplt].set_yticks([0,0.01,0.1])
    #    axes[iplt].set_yticklabels([0,0.01,0.1])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'Feature ranks, normalized')
axes[0].legend(title='width',loc='lower left')
plt.savefig(plot_path+f'featureranks_normalized_percentile{percentile_pcs[ipercentile]}_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()


# ||W||_* / ||\Delta W||_*
plt.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=num_plots,figsize=(num_plots*onefigsize[0],1*onefigsize[1]))
for iplt in range(num_plots):
    for ikey,key in enumerate(sorted(mlp_stats, key=lambda key: int(key[0]))):
        thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats[key][f'delta_W_spectral_norms'])[:,:,iplt]/np.array(mlp_stats[key][f'W_spectral_norms'])[:,:,iplt],mlp_iters[key][f'W_spectral_norms'], fromiter, toiter,zero_init=False)
        if iplt == 2:
            axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION} {key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
        else:
            axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]))
            if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color1, amount=brightnesses[::-1][ikey]),alpha=0.4)
    if compare:
        for ikey,key in enumerate(sorted(mlp_stats2, key=lambda key: int(key[0]))):
            thesedata, lower, upper, iters = get_these_data(np.array(mlp_stats2[key][f'delta_W_spectral_norms'])[:,:,iplt]/np.array(mlp_stats2[key][f'W_spectral_norms'])[:,:,iplt],mlp_iters2[key][f'W_spectral_norms'], fromiter, toiter,zero_init=False)
            if iplt == 2:
                axes[iplt].plot(iters, thesedata, label=f'{PARAMETERIZATION2} {key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
            else:
                axes[iplt].plot(iters, thesedata, label=f'{key[0]}',color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]))
                if fillbetween: axes[iplt].fill_between(iters, lower,upper,color=adjust_lightness(color2, amount=brightnesses[::-1][ikey]),alpha=0.4)
    axes[iplt].set_xlabel('Batches of training')
    if logscale: axes[iplt].set_xscale('symlog')
    #if ylogscale: axes[iplt].set_yscale('symlog')
    axes[iplt].set_title(titles[iplt])
#axes[2].set_ylim(2,axes[2].get_ylim()[1])
axes[0].set_ylabel(r'$\|\Delta W\|_*/\|W\|_*$')
plt.legend(title='width',loc='upper right')
plt.savefig(plot_path+f'deltaW_overW_spectral_{"resnet" if resnet else "mlp"}_{f"over{N_EPOCHS}" if toiter is None else f"{fromiter}iter{toiter}"}_{"ylog" if ylogscale else ""}{"_bands" if fillbetween else ""}_{OPTIM_ALGO}_{PARAMETERIZATION}_{PERTURBATION}_{PARAMETERIZATION2}_{PERTURBATION2}_{SEEDS[0]}.{"png" if fillbetween else "pdf"}',dpi=(500 if fillbetween else 300))
plt.clf()
plt.close('all')


# %%
