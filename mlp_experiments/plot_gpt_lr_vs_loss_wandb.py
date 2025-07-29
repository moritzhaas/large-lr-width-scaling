# %%
'''
Plot the (width-scaled) learning rate versus loss for all widths (the darker, the wider) for GPT.
Requires pre-computed statistics wandb.
'''

#!pip install wandb --upgrade
import os
import numpy as np
from wandb import Api
from utils import mysave, myload
api = Api()

stat_folder = 'stats/litgpt/'
exp_name =  'SGD' #'spfullalign' #'SGD_nogradnorm' #'SGD' # 'varying_warmup' # 'qknorm' # 'qknorm_fixed' # 'mup'
stat_name = f'final_losses_pythia_standard_{exp_name}_width_lr.txt' # 'final_losses_pythia_standard_width_lr_warmup.txt'

if os.path.exists(stat_folder+stat_name):
    results_train,results_val = myload(stat_folder+stat_name)
else:
    results_val, results_train, max_iters = {}, {}, {}
    # For all runs in a project
    if exp_name == 'qknorm':
        runs = api.runs("mup_limitations/standard-transformer-lr-sweep")
    elif exp_name == 'varyingwarmup':
        runs = api.runs("mup_limitations/pretrain-pythia-14m")
    elif exp_name == 'SGD':
        runs = api.runs("mup_limitations/SGD-lr-sweep")
    elif exp_name == 'SGD_nogradnorm':
        runs = api.runs("mup_limitations/SGD-lr-sweep-no-gradient-clipping")
    elif exp_name == 'qknorm_fixed':
        runs = api.runs("mup_limitations/standard-transformer-lr-sweep-layernorm-no-affine")
    elif exp_name == 'spfullalign':
        runs = api.runs("mup_limitations/mup-sp-init-lr-sweep")
    elif exp_name == 'mup':
        runs = api.runs("mup_limitations/mup-lr-sweep")
        
    for run in runs:
        # Get config for this run
        config = run.config
        
        # Get final loss for this run
        if not 'out_dir' in config:
            continue
        
        if exp_name == 'varyingwarmup':
            condition = 'moritz-pythia-14m-standard-width=' in config['out_dir']
        else:
            condition = 'warmup=700-id=2025' in config['out_dir']
        if condition:
            out_dir = config['out_dir']
            try:
                width = int(out_dir.split('width=',2)[1].split('-',2)[0])
                lr = float(out_dir.split('lr=',2)[1].split('-warmup',2)[0])
                warmup = int(out_dir.split('warmup=',2)[1].split('-',2)[0])
            except IndexError: continue

            try:
                history = run.scan_history(keys=["val_loss"])
                final_val_loss = list(history)[-1]["val_loss"]
            except IndexError: continue
            history = run.scan_history(keys=["loss"])
            final_train_loss = list(history)[-1]["loss"]
            lr2 = 10*config['train']['min_lr']
            assert np.abs(lr-lr2)<1e-10, f'LR missmatch {lr} vs {lr2}'
            
            print(f"\nWand Name {run.name}:")
            print(f"Learning Rate, Width, Warmup: {lr}, {width}, {warmup}")
            print(f"Command line args: {config}")
            print(f"Final validation loss: {final_val_loss}")
            print(f"Final training loss: {final_train_loss}")
            if (width,lr,warmup) in results_val.keys():
                if config['eval']['max_iters'] > max_iters[(width,lr,warmup)]:
                    print(f'Overwriting: {width},{lr},{warmup}. Previously {max_iters[(width,lr,warmup)]} iters, now {config["train"]["max_iters"]} iters')
                    results_val[(width,lr,warmup)] = [final_val_loss]
                    results_train[(width,lr,warmup)] = [final_train_loss]
                elif config['eval']['max_iters'] == max_iters[(width,lr,warmup)]:
                    print(f'extending : {width},{lr},{warmup}.')
                    results_val[(width,lr,warmup)].append(final_val_loss)
                    results_train[(width,lr,warmup)].append(final_train_loss)
                else:
                    print(f'Not overweiting: {width},{lr},{warmup}. Previously {max_iters[(width,lr,warmup)]} iters, now {config["train"]["max_iters"]} iters')
            else:
                results_val[(width,lr,warmup)] = [final_val_loss]
                results_train[(width,lr,warmup)] = [final_train_loss]
            max_iters[(width,lr,warmup)] = config['eval']['max_iters']
    mysave(stat_folder,stat_name,[results_train,results_val])


# plot lr vs validation loss for varying width (the darker, the wider) and varying warmup (the more liney the more warmup)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import find, load_multiseed_stats, myload, mysave
from utils.plot_utils import adjust_lightness, adjust_fontsize
#from utils.ssh_utils import get_recent_files_via_ssh, read_hdf5_keys_via_ssh, read_hdf5_entry_via_ssh, get_stats_from_h5py_via_ssh_old, get_recent_folders_via_ssh, establish_ssh_connection, execute_with_retries
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update({'text.usetex': False})
onefigsize = bundles.icml2022()['figure.figsize']

plot_path = './figures/litgpt_loss/'
adjust_fontsize(3.5)
noylabel = False #True
nolegend = False # True

WIDTHS, LRS, WARMUPS=[],[],[]
for key in sorted(results_val, key=lambda key: int(key[0])):
    if key[0] not in WIDTHS: WIDTHS.append(key[0])
    if key[1] not in LRS: LRS.append(key[1])
    if key[2] not in WARMUPS: WARMUPS.append(key[2])
        
WIDTHS, LRS, WARMUPS = np.sort(WIDTHS),np.sort(LRS),np.sort(WARMUPS)
print(WIDTHS,LRS,WARMUPS)

brightnesses= np.linspace(0.5,1.75,len(WIDTHS))
linestyles=['o-','o--','o:'] #intermediate: 'o-.',

lrexp = -0.5 #if exp_name != 'SGD' else 0.0
if lrexp==-1 and exp_name == 'varying_warmup': lrexp = -0.5
if exp_name in ['spfullalign', 'mup']: lrexp = 0
if exp_name == 'qknorm_fixed': lrexp = -1
optlrs_val,optlrs_train = {},{}
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for iwid, WIDTH in enumerate(WIDTHS):
    for iwarm, WARMUP in enumerate(WARMUPS):
        these_stats, lower, upper = {}, {}, {}
        for key in results_val:
            if key[0]==WIDTH and key[2]==WARMUP:
                these_stats[key[1]] = np.mean(results_val[key])   
                if len(results_val[key])>1:
                    thislower = np.percentile(results_val[key], 2.5)
                    thisupper = np.percentile(results_val[key], 97.5)
                else: thislower,thisupper = np.nan,np.nan
                #if thislower is not None and len(thislower)>1: raise ValueError('thislower len>1')
                lower[key[1]] = thislower
                upper[key[1]] = thisupper
        these_stats = dict(sorted(these_stats.items()))
        lower = dict(sorted(lower.items()))
        upper = dict(sorted(upper.items()))
        these_lr = np.array(list(these_stats.keys()))
        print(WIDTH, WARMUP, these_lr, list(these_stats.values()))
        axes.plot(these_lr*(WIDTH/256)**(-lrexp), list(these_stats.values()),linestyles[iwarm],label=f'{WARMUP}' if iwid == len(WIDTHS)-1 else None,color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.75)
        axes.fill_between(these_lr*(WIDTH/256)**(-lrexp), list(lower.values()), list(upper.values()),color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.4)
        if len(list(these_stats.values()))>0:
            optlrs_val[(WIDTH,WARMUP)] = list(these_stats.keys())[np.argmin(list(these_stats.values()))]
axes.set_xscale('log')
axes.set_xlabel(f'Max LR * (width/256)**{-lrexp}')
axes.set_ylabel('Final Valid. Loss')
if exp_name == 'qknorm':
    title = 'SP, ADAM, train LN'
elif exp_name == 'qknorm_fixed':
    title = 'SP, ADAM, fixed LN'
elif exp_name == 'varyingwarmup':
    title = 'SP, ADAM, no qk-layernorm'
elif exp_name == 'SGD':
    title = 'SP, SGD' #, gradient clipping'
elif exp_name == 'SGD_nogradnorm':
    title = 'SP, SGD, no gradient clipping'
elif exp_name == 'spfullalign':
    title = 'SP-full-align, ADAM, qk-layernorm'
elif exp_name == 'mup':
    title = 'mup, ADAM, qk-layernorm'
axes.set_title(title)
color_handles = [mpl.lines.Line2D([0], [0], color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]), label=WIDTH) for iwid, WIDTH in enumerate(WIDTHS)]
legend2 = axes.legend(handles=color_handles,title='width', loc='upper center',frameon=True)
legend2.set_bbox_to_anchor((0.6, 1.0))  # Placed right of the first legend (no overlap)
axes.add_artist(legend2)
if exp_name=='varyingwarmup':
    line_handles = [mpl.lines.Line2D([0], [0], color='tab:blue', linestyle=linest.replace('o',''), label=WARMUPS[il]) for il, linest in enumerate(linestyles)]
    legend1 = axes.legend(handles=line_handles,title='warmup', loc='upper center')
    legend1.set_bbox_to_anchor((0.4, 1.0))  # placed left of upper center
    axes.add_artist(legend1)

#plt.legend(title='warmup', fontsize=8, markerscale=1.2, loc='best')
plt.savefig(plot_path + f'valloss_lr_varying_width_{exp_name}_sp_lrexp{lrexp}_largefont.png',dpi=300)
plt.clf()

print('first plot done')

# same for training error
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
for iwid, WIDTH in enumerate(WIDTHS):
    for iwarm, WARMUP in enumerate(WARMUPS):
        these_stats,lower,upper = {}, {}, {}
        for key in results_train:
            if key[0]==WIDTH and key[2]==WARMUP:
                these_stats[key[1]] = np.mean(results_train[key])
                if len(results_train[key])>1:
                    thislower = np.percentile(results_train[key], 2.5)
                    thisupper = np.percentile(results_train[key], 97.5)
                else: thislower,thisupper = np.nan,np.nan
                #if thislower is not None and len(thislower)>1: raise ValueError('thislower len>1')
                lower[key[1]] = thislower
                upper[key[1]] = thisupper
        these_stats = dict(sorted(these_stats.items()))
        lower = dict(sorted(lower.items()))
        upper = dict(sorted(upper.items()))
        these_lr = np.array(list(these_stats.keys()))
        axes.plot(these_lr*(WIDTH/256)**(-lrexp), list(these_stats.values()),linestyles[iwarm],label=f'{WARMUP}' if iwid == len(WIDTHS)-1 else None,color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.75)
        axes.fill_between(these_lr*(WIDTH/256)**(-lrexp), list(lower.values()), list(upper.values()),color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.4)
        if len(list(these_stats.values()))>0:
            optlrs_train[(WIDTH,WARMUP)] = list(these_stats.keys())[np.argmin(list(these_stats.values()))]
axes.set_xscale('log')
axes.set_xlabel(f'Max LR * (width/256)**{-lrexp}')
axes.set_ylabel('Final Training Loss')

axes.set_title(title)
color_handles = [mpl.lines.Line2D([0], [0], color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]), label=WIDTH) for iwid, WIDTH in enumerate(WIDTHS)]
legend2 = axes.legend(handles=color_handles,title='width', loc='upper center',frameon=True)
legend2.set_bbox_to_anchor((0.6, 1.0))  # Placed right of the first legend (no overlap)
if exp_name=='varyingwarmup':
    line_handles = [mpl.lines.Line2D([0], [0], color='tab:blue', linestyle=linest.replace('o',''), label=WARMUPS[il]) for il, linest in enumerate(linestyles)]
    legend1 = axes.legend(handles=line_handles,title='warmup', loc='upper center')
    legend1.set_bbox_to_anchor((0.4, 1.0))  # placed left of upper center
    axes.add_artist(legend1)
axes.add_artist(legend2)
plt.savefig(plot_path + f'trainloss_lr_varying_width_{exp_name}_sp_lrexp{lrexp}_largefont.png',dpi=300)
plt.clf()


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
lowest = 1000
for iwid, WIDTH in enumerate(WIDTHS):
    for iwarm, WARMUP in enumerate(WARMUPS):
        these_stats,lower,upper = {}, {}, {}
        for key in results_val:
            if key[0]==WIDTH and key[2]==WARMUP:
                these_stats[key[1]] = np.mean(results_val[key])
                if len(results_val[key])>1:
                    thislower = np.percentile(results_val[key], 2.5)
                    thisupper = np.percentile(results_val[key], 97.5)
                else: thislower,thisupper = np.nan,np.nan
                lower[key[1]] = thislower
                upper[key[1]] = thisupper
        these_stats = dict(sorted(these_stats.items()))
        lower = dict(sorted(lower.items()))
        upper = dict(sorted(upper.items()))
        these_lr = np.array(list(these_stats.keys()))
        axes.plot(these_lr*(WIDTH/256)**(-lrexp), list(these_stats.values()),linestyles[iwarm],label=f'{WARMUP}' if iwid == len(WIDTHS)-1 else None,color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.75)
        axes.fill_between(these_lr*(WIDTH/256)**(-lrexp), list(lower.values()), list(upper.values()),color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.4)
        if len(list(these_stats.values()))>0 and np.min(list(these_stats.values())) < lowest:
            lowest = np.min(list(these_stats.values()))
axes.set_xscale('log')
axes.set_xlabel(f'Max LR * (width/256)**{-lrexp}')
axes.set_ylabel('Final Valid. Loss')
axes.set_title(title)
if exp_name in ['varyingwarmup']:
    axes.set_ylim(0.98*lowest,4.3)
elif 'qknorm' in exp_name or 'fullalign' in exp_name or 'mup' in exp_name:
    axes.set_ylim(3.3,4.5)
elif exp_name == 'SGD':
    axes.set_ylim(3.99,7.01)#(0.98*lowest,7.2)
elif exp_name == 'SGD_nogradnorm':
    axes.set_ylim(0.98*lowest,7.2)
if 'qknorm' not in exp_name:
    color_handles = [mpl.lines.Line2D([0], [0], color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]), label=WIDTH) for iwid, WIDTH in enumerate(WIDTHS)]
    legend2 = axes.legend(handles=color_handles,title='width', loc='lower left',frameon=True)
    axes.add_artist(legend2)
if exp_name=='varyingwarmup':
    line_handles = [mpl.lines.Line2D([0], [0], color='tab:blue', linestyle=linest.replace('o',''), label=WARMUPS[il]) for il, linest in enumerate(linestyles)]
    legend1 = axes.legend(handles=line_handles,title='warmup', loc='upper center')
    legend1.set_bbox_to_anchor((0.08, 0.38))  # placed left of upper center
    axes.add_artist(legend1)
plt.savefig(plot_path + f'valloss_lr_varying_width_{exp_name}_sp_lrexp{lrexp}_zoomed_largefont.png',dpi=300)
plt.clf()

# same for training error
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
lowest = 1000
for iwid, WIDTH in enumerate(WIDTHS):
    for iwarm, WARMUP in enumerate(WARMUPS):
        these_stats,lower,upper = {}, {}, {}
        for key in results_train:
            if key[0]==WIDTH and key[2]==WARMUP:
                these_stats[key[1]] = np.mean(results_train[key])
                if len(results_train[key])>1:
                    thislower = np.percentile(results_train[key], 2.5)
                    thisupper = np.percentile(results_train[key], 97.5)
                else: thislower,thisupper = np.nan,np.nan
                #if thislower is not None and len(thislower)>1: raise ValueError('thislower len>1')
                lower[key[1]] = thislower
                upper[key[1]] = thisupper
        these_stats = dict(sorted(these_stats.items()))
        lower = dict(sorted(lower.items()))
        upper = dict(sorted(upper.items()))
        these_lr = np.array(list(these_stats.keys()))
        axes.plot(these_lr*(WIDTH/256)**(-lrexp), list(these_stats.values()),linestyles[iwarm],label=f'{WARMUP}' if iwid == len(WIDTHS)-1 else None,color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.75)
        axes.fill_between(these_lr*(WIDTH/256)**(-lrexp), list(lower.values()), list(upper.values()),color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]),alpha=0.4)
        if len(list(these_stats.values()))>0 and np.min(list(these_stats.values())) < lowest:
            lowest = np.min(list(these_stats.values()))

axes.set_xscale('log')
axes.set_xlabel(f'Max LR * (width/256)**{-lrexp}')
if not noylabel:
    axes.set_ylabel('Final Train Loss')

axes.set_title(title)
if exp_name in ['varyingwarmup']:
    axes.set_ylim(0.98*lowest,4.3)
elif 'qknorm' in exp_name or 'fullalign' in exp_name or 'mup' in exp_name:
    axes.set_ylim(3.3,4.5)
elif exp_name == 'SGD':
    axes.set_ylim(3.99,7.01)#(0.98*lowest,7.2)
elif exp_name == 'SGD_nogradnorm':
    axes.set_ylim(0.98*lowest,7.2)
else:#if exp_name == 'SGD':
    axes.set_ylim(0.98*lowest,7.2)

if not nolegend and 'qknorm' not in exp_name:
    color_handles = [mpl.lines.Line2D([0], [0], color=adjust_lightness('tab:blue',brightnesses[::-1][iwid]), label=WIDTH) for iwid, WIDTH in enumerate(WIDTHS)]
    legend2 = axes.legend(handles=color_handles,title='width', loc='lower left',frameon=True)
    axes.add_artist(legend2)
if exp_name=='varyingwarmup':
    line_handles = [mpl.lines.Line2D([0], [0], color='tab:blue', linestyle=linest.replace('o',''), label=WARMUPS[il]) for il, linest in enumerate(linestyles)]
    legend1 = axes.legend(handles=line_handles,title='warmup', loc='upper center')
    legend1.set_bbox_to_anchor((0.08, 0.38))  # placed left of upper center
    axes.add_artist(legend1)
plt.savefig(plot_path + f'trainloss_lr_varying_width_{exp_name}_sp_lrexp{lrexp}_zoomed_largefont{"_noylabel" if noylabel else ""}.png',dpi=300)
plt.clf()

# %%
