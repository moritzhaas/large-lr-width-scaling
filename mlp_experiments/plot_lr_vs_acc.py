# %%
"""
Plot (width-scaled) learning rate versus accuracy for all widths (the darker, the wider)
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
plot_heatmap = False
few_lines = True
no_zoom = True #False
fill = True #True
log_yscale = True
adjust_fontsize(3)

# the different losses, optimizers and datasets are differentiated by different random seeds. For example:

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


# For example, the comparison between SP-full-align and SP-full-align with increased last-layer learning rate on the generated multi-index model was run as follows:
# lrs_teacher=(0.000316227766 0.001 0.00316227766 0.01 0.0316227766 0.1 0.316227766 0.56234133 1.0 3.16227766 10.0 31.6227766 100.0 316.227766 1000.0)
# for lr in "${lrs_teacher[@]}"; do
#     srun python3 main_mlp_allwidths.py --seed 700 --dataset teacher_ce --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit --optim SGD --nepochs 1 --nomultipliers
#     srun python3 main_mlp_allwidths.py --seed 710 --dataset teacher --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit --optim SGD --nepochs 1 --nomultipliers
#     srun python3 main_mlp_allwidths.py --seed 750 --dataset teacher_ce --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit_largelr --optim SGD --nepochs 1 --nomultipliers
#     srun python3 main_mlp_allwidths.py --seed 760 --dataset teacher --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit_largelr --optim SGD --nepochs 1 --nomultipliers
# done

# lrs_cifar_adam=(0.00001 0.000031622777 0.0001 0.00031622777 0.001 0.0031622777 0.01 0.031622777 0.1 0.31622777 1.0 3.1622777 10.0)
# for lr in "${lrs_cifar_adam[@]}"; do
#     srun python3 main_mlp_allwidths.py --seed 720 --dataset teacher_ce --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit --optim ADAM --nepochs 1 --nomultipliers #--width_choice few
#     srun python3 main_mlp_allwidths.py --seed 730 --dataset teacher --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit --optim ADAM --nepochs 1 --nomultipliers #--width_choice few
#     srun python3 main_mlp_allwidths.py --seed 770 --dataset teacher_ce --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit_largelr --optim ADAM --nepochs 1 --nomultipliers #--width_choice few
#     srun python3 main_mlp_allwidths.py --seed 780 --dataset teacher --nhiddenlayers 7 --lr $lr --rho 0 --param mup_spllit_largelr --optim ADAM --nepochs 1 --nomultipliers #--width_choice few
# done

dataset = 'mnist'  # 'cifar10' # 'cifar10_mse' # 'cifar10_mse_softmax' # 'teacher' # 'teacher_ce' # 'teacher_softmax' # 'mnist' # 'mnist_mse'

if 'cifar10' in dataset:
    stat_path = './stats/cifar10/'
    SEEDS = np.arange(5321,5325) # 8layer and 10 layer fine multiseed SGD
    if 'mse' in dataset:
        SEEDS = np.arange(1010,1012) # cifar10_mse 8layers sgd sp
        #SEEDS = np.arange(1110,1112) # cifar10_mse 8layers adam sp
elif 'mnist' in dataset:
    stat_path = './stats/mnist/'
    SEEDS = np.arange(4321,4325) # 8layer and 10 layer fine multiseed SGD
    if 'mse' in dataset:
        SEEDS = np.array([990,991]) # 8layers sp sgd mse
        #SEEDS = np.array([900,901]) # 8layers sp adam mse
else:
    stat_path = './stats/teacher/'
    SEEDS = np.arange(371,377) # ntrain=10000, balanced classes



epoch = 1 # -1

# add the random seed to the correct case below. Compatibility with the config file will be ensured.
residual = False
if SEEDS[0] in [190, 1293,2293,191,1000,4000,4100,1000,1100,2000,2100]:
    param = 'mup'
elif SEEDS[0] in [191,919,3293,929,292,600,620,610,630,700,720,710,730]:
    param='mup_spllit'
elif SEEDS[0]==192:
    param = 'mup_largelr'
elif SEEDS[0]==2111 or SEEDS[0]==2121:
    param = 'llm'
elif SEEDS[0]==3111 or SEEDS[0]==3121:
    param = 'sp_largeinput'
elif SEEDS[0] in [193,293,2293,650,670,660,680,750,770,760,780]:
    param='mup_spllit_largelr'
else:
    param = 'sp' # 'sp' # 'llm'

# Adam random seeds (else SGD)
SEEDS_ADAM = [215, 271, 272, 273,275,295, 291,471,491, 971,975,1110, 1111,2111,3111,1293,2293,3293,620,630,670,680,720,730,770,780,4100,2100,1100,900]

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

brightnesses= np.linspace(0.5,1.75,2)


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
                if 'cifar10' in filename and len(temp_stats[key]['epoch']) < 21 and SEEDS[0]==90: continue
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
                min_eps = np.min(num_eps)
                for thiskey in all_stats:
                    for subkey in all_stats[key]:
                        all_stats[key][subkey] = [elem[:min_eps] for elem in all_stats[key][subkey]]
                #raise ValueError(f'Epochs in different runs dont coincide for {key}: {num_eps}\n {all_configs[key]}')
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


# all in a single plot:
# light = random features, dark: full training
# '-': relu, ':': linear
# color: number of layers (blue = 1, orange = 2)

# also plot instability threshold
# standard experiments
experiments =[(1, 'relu', 'SGD'), (1, 'relu', 'LL-SGD'), (2, 'relu', 'SGD'), (1, 'linear', 'SGD'), (1, 'linear', 'LL-SGD'), (2, 'linear', 'SGD'), (3, 'relu', 'SGD'),(5, 'relu', 'SGD'), (5, 'linear', 'SGD'), (7, 'relu', 'SGD'),(9, 'relu', 'SGD')]

# depth scaling, seed 390
if SEEDS[0]==390:
    experiments =[(2, 'relu', 'SGD'), (3, 'relu', 'SGD'), (5, 'relu', 'SGD'), (7, 'relu', 'SGD'), (9, 'relu', 'SGD')]

#mnist
if 'mnist' in dataset:
    if SEEDS[0]>=600 and SEEDS[0]<700:
        experiments = [(7,'relu','SGD')]
    else:
        experiments =[(1, 'relu', 'SGD'), (1, 'relu', 'LL-SGD'), (2, 'relu', 'SGD'), (1, 'linear', 'SGD'), (1, 'linear', 'LL-SGD'), (2, 'linear', 'SGD'), (3, 'relu', 'SGD'),(3, 'linear', 'SGD'), (5, 'relu', 'SGD'), (5, 'linear', 'SGD'),(7, 'relu', 'SGD'),(9, 'relu', 'SGD'),(7, 'linear', 'SGD'),(9, 'linear', 'SGD'),]
if 'teacher' in dataset:
    if SEEDS[0]>=700 and SEEDS[0]<800:
        experiments = [(7,'relu','SGD')]

if SEEDS[0] in [990,971,271,4321,5321]:
        experiments = [(7,'relu','SGD'),(9,'relu','SGD')]


#experiments =[(2, 'relu', 'ADAM'),(3, 'relu', 'ADAM'), (5, 'relu', 'ADAM')]

for stattype in ['trainacc','valacc']:
    #if stattype == 'valacc': continue
    if stattype== 'valacc':
        plot_path = 'figures/2dplots/valacc/'
    else:
        plot_path = 'figures/2dplots/'
    if stattype == 'trainacc':
        #if 'teacher' not in stat_path:
        thisstat = 'Loss/train'
        #else:
        #    thisstat = 'Loss/optim'
    else:
        thisstat = 'Loss/val'
    for this_experiment in experiments:
        #for nlay in [1,2]:
        #for activ in ['relu', 'linear']:
        #for optim_algo in ['SGD', 'LL-SGD']:
        #if nlay!=2 or activ != 'linear': continue
        nlay, activ, optim_algo = this_experiment
        
        identifier = f'{nlay+1} layers{" linear" if activ == "linear" else ""}{", rf" if "LL" in optim_algo else ""}'
        filter_strings = [f'stats_mlp*seed{SEED}_*' for SEED in SEEDS]

        if SEEDS[0]==491:
            optim_algo = 'NF-'+optim_algo
        if SEEDS[0] in SEEDS_ADAM:
            optim_algo = optim_algo.replace('SGD','ADAM')

        filenames=[]
        for filter_string in filter_strings:
            for filenam in find(filter_string, stat_path):
                #config = myload(filenam.replace('stats_','config_'))
                if check_compatibility(filenam.replace('stats_','config_'), dataset=dataset, n_hiddenlayers = nlay, activ=activ if activ != 'relu' else None, optim_algo = optim_algo, residual = residual,param=param):
                    filenames.append(filenam)
        #print(dataset, nlay, activ, optim_algo, residual,param)
        if len(filenames) == 0:
            print(f'No files found for {this_experiment},  '+filter_strings[0])
            continue

        mlp_stats, hps = load_multiseed_stats(filenames)

        if mlp_stats is None or len(mlp_stats.keys()) == 0:
            print(f'No stats found for '+identifier)
            continue


        WIDTHS, LRS, RHOS=[],[],[]
        for key in sorted(mlp_stats, key=lambda key: int(key[0])):
            if key[0] not in WIDTHS: WIDTHS.append(key[0])
            if key[1] not in LRS: LRS.append(key[1])
            if key[2] not in RHOS: RHOS.append(key[2])
                
        WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
        #print(WIDTHS,LRS,RHOS)
        ll_stats = {}
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

        train_losses = -9999 * np.ones((len(WIDTHS),len(LRS)))
        lower = -9999 * np.ones((len(WIDTHS),len(LRS)))
        upper = -9999 * np.ones((len(WIDTHS),len(LRS)))

        for i, WIDTH in enumerate(WIDTHS):
            for j, LR in enumerate(LRS):
                    if (WIDTH,LR) in ll_stats.keys():
                        train_losses[i,j] = np.mean(ll_stats[(WIDTH, LR)][:,epoch])
                        thislower, thisupper = np.percentile(ll_stats[(WIDTH, LR)][:,epoch], [2.5, 97.5])
                        lower[i,j] = thislower
                        upper[i,j] = thisupper
                    else:
                        train_losses[i,j]=np.nan
                        lower[i,j], upper[i,j] = np.nan, np.nan


        if log_yscale:
            lower = 100-lower
            upper = 100-upper

        plot_dict = {f'{stattype}_heatmap_widvslr_epoch{epoch}_{dataset}_{param}_seed{SEEDS[0]}-'+identifier.replace(' ','').replace(',','')+'.png': train_losses,}
        for plotname in plot_dict: break
        if plot_heatmap:
            for plotname in plot_dict:
                fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
                vmax = min(np.nanmax(plot_dict[plotname]), 100)
                
                if dataset == 'teacher':
                    label = 'Train MSE' if 'train' in plotname else 'Test MSE'
                    vmin = np.nanmin(plot_dict[plotname])
                    vmax = min(100, vmin*10)
                    cmap = "rocket_r"
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                else:
                    vmin = vmax-10 if 'train' in plotname else vmax - 5 #vmax-4 if 'trainloss' in plotname else vmax - 2
                    label = 'Training accuracy' if 'train' in plotname else 'Test accuracy'
                    cmap= 'rocket'
                    norm=None
                print(plot_dict[plotname].shape,np.sum(~np.isnan(plot_dict[plotname])))
                im = sns.heatmap(plot_dict[plotname][:, ::-1].T,vmin=vmin,vmax=vmax, linewidth=0.5,xticklabels=WIDTHS,yticklabels=np.round(LRS[::-1],4),ax=ax,cbar_kws={'label': label},annot=True,cbar=True,cmap=cmap,norm=norm)
                ax.set_title(identifier+'; '+dataset.replace('_',' '))
                ax.set_ylabel('Learning rate')#r'Learning rate $\eta$')
                ax.set_xlabel('Width')#r'Perturbation radius $\rho$')
                plt.savefig(plot_path + plotname,dpi=300)
                plt.clf()
        
        # plot lr vs acc with prescribed scaling:
        fig,ax=plt.subplots(1,1)
        brightnesses = np.linspace(0.5,1.75,len(WIDTHS))
        if 'teacher' == dataset:
            train_losses = -train_losses
        ioptlr = np.argmax(train_losses,axis=1)
        exp1 = exponent((WIDTHS[0],WIDTHS[-1]), (LRS[ioptlr[0]],LRS[ioptlr[-1]]))
        exponents = np.array([-1,-0.5,-0])
        lrexp = exponents[np.argmin(np.abs(exp1-exponents))]
        # you may want to select the best-fitting exponent, depending on whether the optimum is clear, or whether the maximal stable learning rate is more interesting
        if SEEDS[0]==171 and nlay==2 and activ=='linear':
            lrexp = -0.5
        elif SEEDS[0]==90 and nlay==1 and activ=='linear' and dataset=='cifar10_mse_softmax':
            lrexp = -0.5
        elif SEEDS[0]==90 and nlay==2 and activ=='relu' and dataset=='cifar10_mse_softmax':
            lrexp = -0.5
        elif SEEDS[0]==90 and nlay>2 and activ=='relu' and dataset=='cifar10':
            lrexp = -0.5
        elif SEEDS[0]==371 and nlay==1 and activ=='relu' and dataset=='teacher_ce' and not 'LL' in optim_algo:
            lrexp = 0.0
        elif SEEDS[0]==371 and nlay==2 and dataset=='teacher_softmax':
            lrexp = -1.0
        elif param == 'mup':
            lrexp = 0
        elif SEEDS[0]==971:
            lrexp = -0.5
        elif SEEDS[0]==271:
            lrexp = -1
        elif SEEDS[0]==275:
            lrexp = -0.5
            if nlay==7 or (nlay==2 and activ=='linear'):
                lrexp = -1
        elif SEEDS[0]==291:
            lrexp = -1
        elif SEEDS[0] == 295 and nlay <= 2 and activ == 'relu':
            lrexp = -0.5
        elif SEEDS[0]==295:
            lrexp = -1
        elif SEEDS[0]==390:
            lrexp= -0.5
        elif SEEDS[0]==191 and nlay in [5] and activ=='relu':
            lrexp = 0
        elif SEEDS[0]==990:
            if nlay in [2,3]:
                lrexp = 0.0
            elif 'mse' in dataset:
                lrexp = -1
            else:
                lrexp=-0.5
        elif SEEDS[0]==293 and nlay == 1:
            lrexp = 0
        elif SEEDS[0] in [2293,3293] and activ == 'relu':
            lrexp = 0
        elif SEEDS[0] in [2293] and activ == 'linear':
            lrexp = -0.5
        elif SEEDS[0] == 471:
            lrexp = -1
        elif SEEDS[0] == 491:
            lrexp = -1
        elif SEEDS[0]==1293:
            lrexp = 0
        elif SEEDS[0]==919:
            lrexp = -0.5
        elif SEEDS[0]==929:
            lrexp = 0
        elif SEEDS[0]==292:
            lrexp = 0
        elif SEEDS[0]==225:
            lrexp = -0.5
        elif SEEDS[0]==680:
            lrexp = -0.5
        elif SEEDS[0] in [4321,5321]:
            lrexp = -0.5
        elif 'mnist' in dataset and SEEDS[0] in [600,610,620,630,650,660,670,680]:
            lrexp = 0
        elif 'teacher' in dataset and SEEDS[0] in [700,710,720,730,750,760,770,780]:
            lrexp = -0.5
        elif SEEDS[0] in [971,975]:
            if nlay <= 2:
                lrexp = -0.5
            else:
                lrexp= -1.0
        elif 'mse' in dataset and 'softmax' not in dataset and SEEDS[0]!=919:
            lrexp=-1.0
        if 'teacher' == dataset:
            train_losses = -train_losses
        for iw, WIDTH in enumerate(WIDTHS):
            if few_lines and WIDTH not in [256,1024,4096,16384]: continue
            these_idcs = ~np.isnan(train_losses[iw,:])
            if log_yscale:
                yplot = 100-train_losses[iw,:][these_idcs]
            else:
                yplot = train_losses[iw,:][these_idcs]
            ax.plot(np.array(LRS)[these_idcs]*(WIDTH/256)**(-lrexp), yplot, label=WIDTH,color = adjust_lightness('tab:blue',brightnesses[::-1][iw]))
            if fill: ax.fill_between(np.array(LRS)[these_idcs]*(WIDTH/256)**(-lrexp),lower[iw,:][these_idcs], upper[iw,:][these_idcs], alpha=0.4, color=adjust_lightness('tab:blue',brightnesses[::-1][iw]))
        ymax = np.nanmax(train_losses)
        #lower = np.nanmax(train_losses)
        if not no_zoom:
            if ymax < 90:
                if 'mse' in dataset or dataset == 'teacher':
                    ax.set_ylim(np.maximum(ymax-30,-0.01*ymax), 1.05*ymax)
                    ax.set_ylim(-0.0001, 0.02)#1.05*ymax)
                else:
                    if SEEDS[0] in [271,5321]:
                        ax.set_ylim(ymax-15, 1.05*ymax)
                    else:
                        ax.set_ylim(ymax-10, 1.05*ymax)
            else:
                if SEEDS[0] in [990]:
                    ax.set_ylim(50, np.minimum(100.1,1.05*ymax))
                else:
                    ax.set_ylim(ymax-7, np.minimum(1.05*ymax,100.2))
            if SEEDS[0] in [600,610,620,630,650,660,670,680]:
                ax.set_ylim(84.9, 100.1)
            if SEEDS[0] in [700,710,720,730,750,760,770,780]:
                ax.set_ylim(79.9, 100.1)
            ymin, ymax = ax.get_ylim()
            x_filtered=[]
            for iw in range(len(WIDTHS)):
                these_idcs = ~np.isnan(train_losses[iw,:])
                y=train_losses[iw,:][these_idcs]
                mask = (y >= ymin) & (y <= ymax)
                x=np.array(LRS)[these_idcs]*(WIDTH/256)**(-lrexp)
                x_filtered.extend(x[mask].tolist())
            ax.set_xlim(0.66*np.nanmin(x_filtered),2*np.nanmax(x_filtered))
        if SEEDS[0]==90 and dataset == 'cifar10_mse' and nlay==1 and 'LL' in optim_algo:
            ax.set_xlim(0.003, 10.5)
        if SEEDS[0] in [190,191,293,1293,2293,3293]:
            if 'val' in stattype:
                ax.set_ylim(30,60)
            if SEEDS[0] in [190,191,293]:
                ax.set_xlim(3e-5,5)
            elif SEEDS[0] in [1293]:
                ax.set_xlim(1e-4, 4)
            elif SEEDS[0] in [2293,3293]:
                ax.set_xlim(1e-5, 5)
        if dataset=='teacher':
            ax.set_ylim(-0.0001, 0.02)#1.05*ymax)

        ax.set_xscale('log')
        ax.set_xlabel(f'LR * (WIDTH/256)**({-lrexp})')
        if stattype=='trainacc':
            thisylabel = 'Training accuracy' if dataset!='teacher' else 'Training MSE'
        else:
            thisylabel = 'Validation accuracy' if dataset!='teacher' else 'Validation MSE'
        if SEEDS[0] != 4321:
            ax.set_ylabel(thisylabel)
        if log_yscale:
            if SEEDS[0] != 4321:
                ax.set_ylabel(r'$100\%-$Train accuracy')
            ax.set_yscale('log')

        if SEEDS[0]==4321:
            ax.set_title('CE SGD MNIST')
            ax.set_xlim(9.99e-5,1.01)
            if not log_yscale:
                ax.set_ylim(39.9,100.01)
            else:
                ax.set_ylim(0.99,100.01)
        elif SEEDS[0]==5321:
            ax.set_title('CE SGD CIFAR10')
            ax.set_xlim(9.99e-5,3.01)
        elif SEEDS[0]==990 and 'mse' in dataset:
            ax.set_title('MSE SGD MNIST')
            if not log_yscale:
                ax.set_ylim(39.9,100.01)
            else:
                ax.set_ylim(0.99,100.01)
            ax.set_xlim(9.99e-5,1.01)
        elif SEEDS[0]==1010 and 'mse' in dataset:
            ax.set_title('MSE SGD CIFAR10')
            ax.set_xlim(9.99e-5,1.01)
        if SEEDS[0]==4321:
            plt.legend(title='width',frameon=True,loc='upper left')
        elif not(SEEDS[0]==990 and 'mse' in dataset):
            plt.legend(title='width',frameon=True)
        plt.savefig(plot_path+f'lrexp{lrexp}_vsacc_{"nozoom_" if no_zoom else ""}{"fill_" if fill else ""}{"logy_" if log_yscale else ""}'+plotname.replace('heatmap_widvslr',''))
        plt.clf()
# %%


