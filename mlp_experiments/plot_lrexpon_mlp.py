# %%
"""
Plot width versus the optimal learning rate and the minimal unstable learning rate for MLPs, and fit scaling laws.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import find, load_multiseed_stats, myload, mysave
from utils.plot_utils import adjust_lightness, adjust_fontsize
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

def get_expon(xs,ys):
    x_log,y_log = np.log10(xs), np.log10(ys)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    return slope, intercept

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


def load_compatible_stats(seeds, loss, optim, param, nhidden=7):
        filter_string_mse = [f'stats_mlp*seed{seed}_*' for seed in seeds]

        filenames_mse=[]
        for filenam in find(filter_string_mse, './stats/'+dataset+'/'):
            #config = myload(filenam.replace('stats_','config_'))
            checkdataset = dataset+'_mse' if loss == 'mse' else dataset
            if check_compatibility(filenam.replace('stats_','config_'), dataset=checkdataset, n_hiddenlayers = nhidden, optim_algo = optim,param=param):
                filenames_mse.append(filenam)
        if len(filenames_mse) == 0:
            print(f'No files found for {seeds[0]}, {dataset}, {optim}, {param}')
            return None, None

        mlp_stats, hps = load_multiseed_stats(filenames_mse)

        return mlp_stats, hps


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

    if mlp_stats is None or len(mlp_stats.keys()) == 0:
        print(f'No stats found')

    WIDTHS, LRS, RHOS=[],[],[]
    for key in sorted(mlp_stats, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS: WIDTHS.append(key[0])
        if key[1] not in LRS: LRS.append(key[1])
        if key[2] not in RHOS: RHOS.append(key[2])
            
    WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)

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
        if dataset == 'teacher': OPTACC = -1e8
        for lr in LRS:
            for key in ll_stats.keys():
                if key[0] == width and key[1] == lr:
                    thisacc = np.mean(ll_stats[key])
                    if dataset == 'teacher': thisacc = -thisacc
                    if thisacc>OPTACC:
                        OPTLR=key[1]
                        OPTACC=thisacc
                    print(f'lr {key[1]}: {thisacc}')

        for lr in LRS:
            for key in ll_stats.keys():
                if key[0] == width and key[1] == lr:
                    thisacc = np.mean(ll_stats[key])
                    if np.isnan(min_acc):
                        if np.isnan(thisacc) and key[1]< MAXLR and key[1]>=OPTLR:
                            MAXLR = key[1]
                    else: 
                        if thisacc<min_acc and key[1]< MAXLR and key[1]>=OPTLR:
                            MAXLR=key[1]
        
        opt_lrs[width] = OPTLR
        opt_accs[width] = OPTACC
        max_lrs[width] = MAXLR if MAXLR<1e6 else np.nan
    
    print(opt_lrs)
    return list(opt_lrs.keys()), list(opt_lrs.values()), list(max_lrs.values())

plot_path = './figures/mainfigures/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    
# %%
# the different losses, optimizers and datasets are differentiated by different random seeds
mse_sp = [371,1010,990] #[1010,1110,990,900]
ce_sp = [371,5321,4321] #[5321,271,4321,971]

epoch = 1
thisstat = 'Loss/train'
nhiddenteacher=1

loss = 'ce'

if loss == 'mse':
    seeds = mse_sp
else:
    seeds = ce_sp

adjust_fontsize(3)
nseeds=2
colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red']#['#1f77b4','#ff7f0e']
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
iline=0

nolegend = (loss == 'ce')


if nhiddenteacher == 1:
    text_pos_mse = [2.0,1.5,0.3]
    text_pos_ce = [0.2,1.7,0.3]
else:
    text_pos_mse = [0.35,1.5,0.35]
    text_pos_ce = [1.8,1.8,0.35]

text_pos_y = {('mse', 371): [0.5,1.7], ('mse', 1010): [0.0007,0.002], ('mse', 990): [0.0007,0.002],('ce', 371): [0.8,2.2], ('ce', 5321): [0.5,1.7], ('ce', 4321): [0.3,1.2],}

for seed in seeds:
    param = 'sp'
    
    if seed in [191,371,390,1000,1010,4000,2000,990,4321,5321]:
        optim = 'SGD'
    else:
        optim = 'ADAM'
    if seed in [191,390,1000,1010,1100,1110,2293,271,5321]:
        dataset = 'cifar10'
    elif seed == 371:
        dataset = 'teacher'
    else:
        dataset = 'mnist'

    if seed in [4321,5321]:
        nseeds = 4
    else:
        nseeds = 2
    if loss=='ce':
        if seed == 371:
            widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed+iseed for iseed in range(6)],'teacher_ce', optim_algo=optim,stat_path = './stats/'+'teacher/', min_acc=54,n_hidden=nhiddenteacher)
        else:
            widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed+iseed for iseed in range(nseeds)],dataset, optim_algo=optim,stat_path = './stats/'+dataset+'/')
    else:
        if seed == 371:
            widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed+iseed for iseed in range(6)],'teacher', optim_algo=optim,stat_path = './stats/'+'teacher'+'/', min_acc=np.nan,n_hidden=nhiddenteacher)
        else:
            widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed+iseed for iseed in range(nseeds)],dataset+'_mse', optim_algo=optim,stat_path = './stats/'+dataset+'/')

    text_wid = 2048 if 'cifar10' not in dataset else 512
    
    if loss== 'ce':
        if 'teacher' in dataset:
            text_wid = 512 #358.4
        elif 'cifar10' in dataset:
            text_wid = 2048 #896
        else:
            text_wid = 8192 #2240

    ypos = text_pos_y[(loss,seed)]
    x_log,y_log = np.log10(widths_mnist), np.log10(lrs_mnist)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    y_log_pred = 10**(slope * (x_log) + intercept)
    axes.plot(widths_mnist, y_log_pred,label= dataset,color=colors[iline])
    axes.scatter(widths_mnist,lrs_mnist,s=12,marker='x', color=colors[iline],)
    axes.text(text_wid,ypos[0],f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14,fontweight='bold')
    
    x_log,y_log = np.log10(widths_mnist), np.log10(max_lrs_mnist)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    y_log_pred = 10**(slope * (x_log) + intercept)
    axes.plot(widths_mnist, y_log_pred,'--',color=colors[iline])
    axes.scatter(widths_mnist,max_lrs_mnist,s=12,marker='o', color=colors[iline],)
    axes.text(text_wid,ypos[1],f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14,fontweight='bold')
    iline += 1

    
    
axes.set_xscale('log')
axes.set_yscale('log')
WIDTHS = [256,1024,4096,16384]
axes.set_xticks(WIDTHS)
axes.set_xticklabels(WIDTHS)
axes.set_xlabel('Width')
if not nolegend:
    axes.set_ylabel('Learning Rate')
axes.set_title('SGD, MSE' if loss=='mse' else 'SGD, CE')
#axes.set_ylim(0.0001,50)
if not nolegend:
    plt.legend(loc='upper right',frameon=True)
# adjust_fontsize(4)
plt.savefig(plot_path + f'mainfig_lrexponents_maxvsopt_{loss}_alldatasets_onlysgd{"_teacherhidden"+str(nhiddenteacher) if nhiddenteacher is not None else ""}.png',dpi=300)
plt.clf()


# %%
maxstable = True

for seed_mse, seed_ce in zip(mse_sp,ce_sp):
    param = 'sp'
    
    if seed_mse in [191,371,390,1000,1010,4000,2000,990,4321,5321]:
        optim = 'SGD'
    else:
        optim = 'ADAM'
    if seed_mse in [191,390,1000,1010,1100,1110,2293,271,5321]:
        dataset = 'cifar10'
    elif seed_mse == 371:
        dataset = 'teacher'
    else:
        dataset = 'mnist'

    if seed_ce in [4321,5321]:
        nseeds = 4
    else:
        nseeds = 2
    if seed_mse == 371:
        widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed_ce+iseed for iseed in range(6)],'teacher_ce', optim_algo=optim,stat_path = './stats/'+'teacher/', min_acc=54,n_hidden=nhiddenteacher)
    else:
        widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed_ce+iseed for iseed in range(nseeds)],dataset, optim_algo=optim,stat_path = './stats/'+dataset+'/')

    if maxstable:
        lrs_mnist = max_lrs_mnist
    x_log,y_log = np.log10(widths_mnist), np.log10(lrs_mnist)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    y_log_pred = 10**(slope * (x_log) + intercept)
    axes.plot(widths_mnist, y_log_pred,label= dataset,color=colors[iline])
    axes.scatter(widths_mnist,lrs_mnist,s=12,marker='x', color=colors[iline],)
    axes.text(2048 if seed_mse == 371 else widths_mnist[-2],lrs_mnist[-2]*text_pos_ce[iline],f'{np.round(slope,2)}',color=colors[iline], ha='center',alpha=0.95,fontsize=12,fontweight='bold')

    if seed_mse == 371:
        widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed_mse+iseed for iseed in range(6)],'teacher', optim_algo=optim,stat_path = './stats/'+'teacher'+'/', min_acc=np.nan,n_hidden=nhiddenteacher)
    else:
        widths_mnist, lrs_mnist, max_lrs_mnist = get_best_lrs_from_stats([seed_mse+iseed for iseed in range(nseeds)],dataset+'_mse', optim_algo=optim,stat_path = './stats/'+dataset+'/')

    if maxstable:
        lrs_mnist = max_lrs_mnist

    x_log,y_log = np.log10(widths_mnist), np.log10(lrs_mnist)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    y_log_pred = 10**(slope * (x_log) + intercept)
    axes.plot(widths_mnist, y_log_pred,'--',color=colors[iline])
    axes.scatter(widths_mnist,lrs_mnist,s=12,marker='o', color=colors[iline],)
    axes.text(widths_mnist[1],lrs_mnist[1]*text_pos_mse[iline],f'{np.round(slope,2)}',color=colors[iline], ha='center',alpha=0.95,fontsize=12,fontweight='bold')
    iline += 1
    
    
axes.set_xscale('log')
axes.set_yscale('log')
WIDTHS = [256,1024,4096,16384]
axes.set_xticks(WIDTHS)
axes.set_xticklabels(WIDTHS)
axes.set_xlabel('Width')
axes.set_ylabel('Min. unstable LR' if maxstable else 'Optimal learning rate')
axes.set_title('SGD, CE (solid) vs MSE (dashed)')
#axes.set_ylim(0.0001,50)
plt.legend(loc='upper right',frameon=True)
# adjust_fontsize(4)
plt.savefig(plot_path + f'mainfig_lrexponents_msevsce_onlysgd{"_maxstable" if maxstable else "_optlr"}{"_teacherhidden"+str(nhiddenteacher) if nhiddenteacher is not None else ""}.png',dpi=300)
plt.clf()


# %%
all_best_losses = {}
for seed_mse, seed_ce in zip(mse_sp,ce_sp):
    param = 'sp'
    
    if seed_mse in [191,390,1000,1010,4000,2000,990]:
        optim = 'SGD'
    else:
        optim = 'ADAM'
    if seed_mse in [191,390,1000,1010,1100,1110,2293,271]:
        dataset = 'cifar10'
    else:
        dataset = 'mnist'

    mlp_stats, hps = load_compatible_stats(seed_mse,'mse',optim, param)
    mlp_stats_ce, hps_ce = load_compatible_stats(seed_ce,'ce',optim, param)
    
    if mlp_stats is None or len(mlp_stats.keys()) == 0:
        print(f'No stats found for {seed_mse}, mse, {dataset}, {optim}, {param}')
        continue
    if mlp_stats_ce is None or len(mlp_stats_ce.keys()) == 0:
        print(f'No stats found for {seed_ce}, ce, {dataset}, {optim}, {param}')
        continue

    WIDTHS, LRS, RHOS=[],[],[]
    for key in sorted(mlp_stats, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS: WIDTHS.append(key[0])
        if key[1] not in LRS: LRS.append(key[1])
        if key[2] not in RHOS: RHOS.append(key[2])
            
    WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
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
            all_best_losses[('mse',optim,param,seed_mse,WIDTH,key[1],dataset)] = np.mean(final_stats[key[1]])

    for WIDTH in WIDTHS:
        final_stats = {}
        for key in mlp_stats_ce:
            if key[0] != WIDTH: continue
            if len(mlp_stats_ce[key][thisstat][0][0])==3:
                for irun in range(len(mlp_stats_ce[key][thisstat])):
                    iters, accs = read_stats(mlp_stats_ce[key][thisstat][irun], epoch_count=True)
                    if key[1] in final_stats:
                        final_stats[key[1]].append(accs[epoch])
                    else:
                        final_stats[key[1]] = [accs[epoch]]
            all_best_losses[('ce',optim,param,seed_ce,WIDTH,key[1],dataset)] = np.mean(final_stats[key[1]])

# %%