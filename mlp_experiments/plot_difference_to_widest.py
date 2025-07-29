# %%
"""
Plot the tracked difference to the widest model as a function of width for a fixed time step
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import get_filename_from_args, mysave, myload, find
from utils.plot_utils import adjust_lightness, width_plot
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"mathtext.fontset": 'cm'})

brightnesses= np.linspace(0.5,1.75,4)
plot_path = './figures/width_plots/' 


OPTIM_ALGO = 'SGD' #'ELEMENTSAM-SGD'
OPTIM_ALGO2 = 'SGD' #'LAYERSAM-SGD' #'SAM-SGD-LL'#'SAM-SGD' # 'LL-SAM-ELSE-SGD' #'SAM-SGD-LL' 'SAM-SGD-NOBN'
OPTIM_ALGO3 = 'SGD'#'SGD'
PARAMETERIZATION = 'mup'
PARAMETERIZATION2 = 'sp'#'mup' #'sp'
PERTURBATION = 'mpp'#'mpp'
PERTURBATION2 = 'naive' #'naive'# None#'first_layer'#'mpp'
SEEDS2=[]
SEEDS3=[]
LL_INIT = False#True #True # 0 init of last layer
EVAL_ITER = 0
resnet = False
if PARAMETERIZATION2 is None: PARAMETERIZATION2=PARAMETERIZATION
if PERTURBATION2 is None: PERTURBATION2=PERTURBATION

N_EPOCHS = 20 # 20 #1
LR_MUP = 0.3162 #52: 0.3162 #'' #None
LR_SP = 0.1 #52: 1.0 #'' #None

SEEDS= [152+i for i in range(4)] #[52,53,54,55]
SEEDS2=SEEDS #[52,53,54,55]
# SGD
#SEEDS3 =[711023]


sgdexists = (len(SEEDS3)>0)

filenames = []
for SEED in SEEDS:
    filenames.extend(find(f'mlp*lr={LR_MUP}*nehs={N_EPOCHS}*paam='+PARAMETERIZATION+'*perb='+PERTURBATION+'*opim='+OPTIM_ALGO+f'*seed={SEED}-*fial-*','stats/cifar10/'))

filenames2 = []
for SEED2 in SEEDS2:
    filenames2.extend(find(f'mlp*lr={LR_SP}*nehs={N_EPOCHS}*paam='+PARAMETERIZATION2+'*perb='+PERTURBATION2+'*opim='+OPTIM_ALGO2+f'*seed={SEED2}-*fial-*','stats/cifar10/'))


def load_multiseed_stats(seeds,optimalgo='',param='',perturb='', lr = '', nepochs=''):
    all_stats={}
    if len(seeds)>0:
        for seed in seeds:
            for ll_filename in find(f'mlp*lr={lr}*nehs={nepochs}*paam='+param+'*perb='+perturb+'*opim='+optimalgo+f'*seed={seed}-*fial-*','stats/cifar10/'):
                # ll_filename = find(f'mlp*paam='+param+'*perb='+perturb+'*opim='+optimalgo+f'*seed={seed}-*','stats/cifar10/')[0]#'stats/scaletracking/'
                n_epochs = np.int64(ll_filename.split('-nehs=',2)[1].split('-',2)[0])
                try:
                    temp_stats = myload(ll_filename)
                except RuntimeError:
                    continue
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
    try:
        return all_stats,n_epochs
    except UnboundLocalError:
        return None, None

ll_stats, N_EPOCHS = load_multiseed_stats(SEEDS,OPTIM_ALGO,PARAMETERIZATION,'',LR_MUP,N_EPOCHS)#myload(ll_filename)
ll_stats2, N_EPOCHS2 = load_multiseed_stats(SEEDS2,OPTIM_ALGO2,PARAMETERIZATION2, '',LR_SP,N_EPOCHS)#myload(ll_filename2)
if sgdexists: sgd_stats, N_EPOCHS3 = load_multiseed_stats(SEEDS3)#myload(sgd_filename)

if N_EPOCHS2 != N_EPOCHS: raise ValueError(f'Differing number of epochs: {N_EPOCHS} vs {N_EPOCHS2}')
N_HIDDEN_LAYERS = 2 #np.int64(ll_filename.split('-nhrs=',2)[1].split('-',2)[0])

# find all widths, lrs, rhos from stat keys:
WIDTHS, LRS, RHOS=[],[],[]
for key in sorted(ll_stats, key=lambda key: int(key[0])):
    if key[0] not in WIDTHS: WIDTHS.append(key[0])
    if key[1] not in LRS: LRS.append(key[1])
    if key[2] not in RHOS: RHOS.append(key[2])
        
WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)
print(WIDTHS,LRS,RHOS)

if len(WIDTHS)>=6:
    WIDTHS= WIDTHS[:5]
    for key in sorted(ll_stats, key=lambda key: int(key[0])):
        if key[0]>8192: del ll_stats[key]

for key in sorted(ll_stats, key=lambda key: int(-key[0])): break
for key2 in sorted(ll_stats2, key=lambda key: int(-key[0])): break
if sgdexists:
    for sgdkey in sorted(sgd_stats, key=lambda key: int(-key[0])): break


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

last_iters, last_stats = process_stats(ll_stats, picklast=True)
iters, ll_stats = process_stats(ll_stats, epoch_count=False)
last_iters2, last_stats2 = process_stats(ll_stats2, picklast=True)
iters2, ll_stats2 = process_stats(ll_stats2, epoch_count=False)

# if not multiple seeds, make sure still same shape
if len(np.array(ll_stats[key]['epoch']).shape)==1:
    for thiskey in ll_stats:
        for subkey in ll_stats[thiskey]:
            ll_stats[thiskey][subkey] = [ll_stats[thiskey][subkey]]

if len(np.array(ll_stats2[key2]['epoch']).shape)==1:
    for thiskey in ll_stats2:
        for subkey in ll_stats2[thiskey]:
            ll_stats2[thiskey][subkey] = [ll_stats2[thiskey][subkey]]

N_RUNS = len(ll_stats[key][f'Loss/train'])
N_RUNS2 = len(ll_stats2[key2][f'Loss/train'])
N_ITER = len(ll_stats[key][f'Loss/train'][0])

for subkey in ll_stats[key].keys():
    if len(ll_stats[key][subkey][0])>0:
        try:
            finalstats = [ll_stats[key][subkey][irun][-1] for irun in range(N_RUNS)]
            meanfinal=np.mean(finalstats)
            print(subkey,meanfinal)
        except:
            continue

#print(np.mean([ll_stats[key]['Loss/val'][irun][-1] for irun in range(len(ll_stats[key]['Loss/val']))]), np.mean([ll_stats2[key2]['Loss/val'][irun][-1] for irun in range(len(ll_stats2[key2]['Loss/val']))]))
#if sgdexists: print(np.mean([sgd_stats[sgdkey]['Loss/val'][irun][-1] for irun in range(len(sgd_stats[sgdkey]['Loss/val']))]))

if PARAMETERIZATION == 'mup' and PARAMETERIZATION2 == 'sp':
    if PERTURBATION != PERTURBATION2:
        label1 = f'{PARAMETERIZATION} {PERTURBATION}'
        label2 = f'{PARAMETERIZATION2} {PERTURBATION2}'
        label3 = f'{PARAMETERIZATION2} {OPTIM_ALGO3}'
    else:
        label1 = f'{PARAMETERIZATION}'
        label2 = f'{PARAMETERIZATION2}'
        label3 = f'{PARAMETERIZATION2} {OPTIM_ALGO3}'
else:
    label1 = OPTIM_ALGO if (PERTURBATION == 'global' or PERTURBATION == PERTURBATION2) else PERTURBATION
    label2 = OPTIM_ALGO2 if(PERTURBATION == 'global' or PERTURBATION == PERTURBATION2) else PERTURBATION2
    label3 = OPTIM_ALGO3



try:
    print('Evaltime vs traintime: ',np.mean(ll_stats[key]['evaltime'],axis=0)[-1],np.mean(ll_stats[key]['traintime'],axis=0)[-1])
except:
    print('Evaltime vs traintime: ',np.mean(ll_stats[key]['evaltime'],axis=0),np.mean(ll_stats[key]['traintime'],axis=0))


# %%
# diff2wide with scaling law
color2 = 'tab:orange'
color1 = 'tab:blue'

# compute exponent
def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    return (np.log(y[0])-np.log(y[1]))/(np.log(x[0])-np.log(x[1]))

def scaling_law(x,y):
    exp1 = exponent(x,y)
    const =  y[0] * x[0]**(-exp1)
    return (lambda inp: const * inp**(exp1))

for subkey in ['l2diff2wide_train','l2diff2wide_test']:

    l2diff=np.array([[ll_stats[key][subkey][irun][-1] for irun in range(N_RUNS)] for key in sorted(ll_stats, key=lambda key: int(key[0]))])
    l2diff2=np.array([[ll_stats2[key][subkey][irun][-1] for irun in range(N_RUNS)] for key in sorted(ll_stats2, key=lambda key: int(key[0]))])

    WIDTHS2 = []
    for key in sorted(ll_stats2, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS2:   WIDTHS2.append(key[0])
    WIDTHS2 = np.sort(WIDTHS2)

    mean_l2diff=np.mean(l2diff, axis=1)
    mean_l2diff2=np.mean(l2diff2,axis=1)

    exp1, exp2 = exponent((WIDTHS[0],WIDTHS[-1]),(mean_l2diff[0],mean_l2diff[-1])), exponent((WIDTHS2[0],WIDTHS2[-1]),(mean_l2diff2[0],mean_l2diff2[-1]))
    scaling_law1, scaling_law2 = scaling_law((WIDTHS[0],WIDTHS[-1]),(mean_l2diff[0],mean_l2diff[-1])), scaling_law((WIDTHS2[0],WIDTHS2[-1]),(mean_l2diff2[0],mean_l2diff2[-1]))

    plt.tight_layout()
    fig,ax = plt.subplots(1,1,figsize=(1*onefigsize[0],1*onefigsize[1]))
    ax.plot(WIDTHS, mean_l2diff, label=PARAMETERIZATION, color=color1)
    ax.fill_between(WIDTHS, np.quantile(l2diff,0.025,axis=1), np.quantile(l2diff,0.975,axis=1), color=color1, alpha=0.4)
    ax.plot((WIDTHS[0],WIDTHS[-1]),(mean_l2diff[0],mean_l2diff[-1]),linestyle='--',color=color1)
    ax.plot(WIDTHS2, mean_l2diff2, label=PARAMETERIZATION2,color=color2)
    ax.fill_between(WIDTHS2, np.quantile(l2diff2,0.025,axis=1), np.quantile(l2diff2,0.975,axis=1), color=color2, alpha=0.4)
    ax.plot((WIDTHS2[0],WIDTHS2[-1]),(mean_l2diff2[0],mean_l2diff2[-1]),linestyle='--',color=color2)
    ax.text(WIDTHS[1],scaling_law1(WIDTHS[1])*0.85,f'{np.round(exp1,3)}',color=color1, ha='center')
    ax.text(WIDTHS2[1],scaling_law2(WIDTHS2[1])*1.15,f'{np.round(exp2,3)}',color=color2, ha='center')
    #ax.set_ylim(0.0001,1)
    ax.set_xlabel('width')
    ax.set_ylabel('L2diff2wide train' if 'train' in subkey else 'L2diff2wide test')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(WIDTHS)
    ax.set_xticklabels(WIDTHS)
    ax.set_xticks([], minor=True)
    ax.legend()
    plt.savefig(plot_path+subkey+f'_epochs{N_EPOCHS}_{SEEDS[0]}_{SEEDS2[0]}.png',dpi=300)
    plt.close('all')


# %%