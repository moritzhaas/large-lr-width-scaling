# %%
"""
Plot width versus the optimal learning rate and the minimal unstable learning rate for GPT, and fit scaling laws.
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils import  myload
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

adjust_fontsize(3)

WIDTHS = [256, 1024, 4096]

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


opt_lrs, opt_accs,max_lrs = {}, {}, {}
for iw, width in enumerate(WIDTHS):
    print('Width ', width)
    OPTLR,OPTLOSS,MAXLR=0,1e8,1e8
    for lr in LRS:
        for key in results_train.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train[key])
                if thisloss<OPTLOSS:
                    OPTLR=key[1]
                    OPTLOSS=thisloss
                print(f'lr {key[1]}: {thisloss}')
    maxloss=OPTLOSS+1
    for lr in LRS:
        for key in results_train.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train[key])
                if thisloss>maxloss and key[1]< MAXLR and key[1]>=OPTLR:
                    MAXLR=key[1]
    
    opt_lrs[width] = OPTLR
    opt_accs[width] = OPTLOSS
    max_lrs[width] = MAXLR if MAXLR<1e6 else np.nan

print(opt_lrs, max_lrs)
            
lrs_gpt_nogradnorm = [0.31622776601683794,0.15811388300841897, 0.07905694150420949]
max_lrs_gpt_nogradnorm = [0.517947467923121, 0.42417144912203586, 0.12948686698078024]
exp_gpt_nogradnorm = -0.5

lrs_gpt= list(opt_lrs.values()) #[5.17947467923121,3.5984283650057587, 1.7992141825028793]
max_lrs_gpt = list(max_lrs.values()) #[14.0, 9.75, 3.5]
exp_gpt = -0.38 #-0.38135971089688003
exp_gpt_max = -0.5

#%%
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

opt_lrs, opt_accs,max_lrs = {}, {}, {}
for iw, width in enumerate(WIDTHS):
    print('Width ', width)
    OPTLR,OPTLOSS,MAXLR=0,1e8,1e8
    for lr in LRS:
        for key in results_train2.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train2[key])
                if thisloss<OPTLOSS:
                    OPTLR=key[1]
                    OPTLOSS=thisloss
                print(f'lr {key[1]}: {thisloss}')
    maxloss=OPTLOSS+1
    for lr in LRS:
        for key in results_train2.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train2[key])
                if thisloss>maxloss and key[1]< MAXLR and key[1]>=OPTLR:
                    MAXLR=key[1]
    
    opt_lrs[width] = OPTLR
    opt_accs[width] = OPTLOSS
    max_lrs[width] = MAXLR if MAXLR<1e6 else np.nan

print(opt_lrs, max_lrs)

lrs_gpt_adam = list(opt_lrs.values()) # [0.01, 0.007905694150420948, 0.00316227766]
max_lrs_adam = list(max_lrs.values()) # [0.9, 0.675, 0.16875]
exp_gpt_adam = -0.42 # exponent(WIDTHS[:3],lrs_gpt_adam) #-0.41524101188012513
exp_gpt_max_adam = -0.60 #-0.6037593748197111

x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
exp_gpt_adam=np.round(slope,2)

x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_adam)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
exp_gpt_max_adam=np.round(slope,2)
print(lrs_gpt_adam, max_lrs_adam, exp_gpt_adam, exp_gpt_max_adam)

results_train2,results_val2 = myload('stats/litgpt/'+f'final_losses_pythia_standard_qknorm_fixed_width_lr.txt')
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

opt_lrs, opt_accs,max_lrs = {}, {}, {}
for iw, width in enumerate(WIDTHS):
    print('Width ', width)
    OPTLR,OPTLOSS,MAXLR=0,1e8,1e8
    for lr in LRS:
        for key in results_train2.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train2[key])
                if thisloss<OPTLOSS:
                    OPTLR=key[1]
                    OPTLOSS=thisloss
                print(f'lr {key[1]}: {thisloss}')
    maxloss=OPTLOSS+1
    for lr in LRS:
        for key in results_train2.keys():
            if key[0] == width and key[1] == lr:
                thisloss = np.mean(results_train2[key])
                if thisloss>maxloss and key[1]< MAXLR and key[1]>=OPTLR:
                    MAXLR=key[1]
    
    opt_lrs[width] = OPTLR
    opt_accs[width] = OPTLOSS
    max_lrs[width] = MAXLR if MAXLR<1e6 else np.nan

print(opt_lrs, max_lrs)

lrs_gpt_adam_fixed = list(opt_lrs.values()) # [0.01, 0.007905694150420948, 0.00316227766]
max_lrs_adam_fixed = list(max_lrs.values()) # [0.9, 0.675, 0.16875]

x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam_fixed)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
exp_gpt_adam_fixed=np.round(slope,2)

x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_adam_fixed)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
exp_gpt_max_adam_fixed=np.round(slope,2)
print(lrs_gpt_adam_fixed, max_lrs_adam_fixed, exp_gpt_adam_fixed, exp_gpt_max_adam_fixed)


# %%

# only GPT plot
plot_path = './figures/mainfigures/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

colors = ['tab:blue','tab:orange', 'tab:green', 'tab:red']

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(1*onefigsize[0],1*onefigsize[1]))
iline=0

# GPT SGD
WIDTHS = [256,1024,4096]
x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,label= 'SGD',color=colors[iline])
axes.scatter(WIDTHS,lrs_gpt,s=12,marker='x', color=colors[iline],)
axes.text(512,10**(slope * np.log10(512) + intercept)*0.35,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')

x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_gpt)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,'--',color=colors[iline])
axes.scatter(WIDTHS,max_lrs_gpt,s=12,marker='o', color=colors[iline],)
axes.text(2048,10**(slope * np.log10(2048) + intercept)*1.5,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
iline += 1

# GPT SGD nogradnorm
# WIDTHS = [256,1024,4096]
# x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_nogradnorm)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
# y_log_pred = 10**(slope * (x_log) + intercept)
# axes.plot(WIDTHS, y_log_pred,label= 'SGD',color=colors[iline])
# axes.scatter(WIDTHS,lrs_gpt_nogradnorm,s=12,marker='x', color=colors[iline],)
# axes.text(512,10**(slope * np.log10(512) + intercept)*0.35,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')

# x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_gpt_nogradnorm)
# slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
# y_log_pred = 10**(slope * (x_log) + intercept)
# axes.plot(WIDTHS, y_log_pred,'--',color=colors[iline])
# axes.scatter(WIDTHS,max_lrs_gpt_nogradnorm,s=12,marker='o', color=colors[iline],)
# axes.text(2048,10**(slope * np.log10(2048) + intercept)*1.5,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
# iline += 1


# GPT LN
WIDTHS = [256,1024,4096]
x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,label= 'train LN',color=colors[iline])
axes.scatter(WIDTHS,lrs_gpt_adam,s=12,marker='x', color=colors[iline],)
axes.text(2048,10**(slope * np.log10(2048) + intercept)*1.5,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')

x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_adam)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,'--',color=colors[iline])
axes.scatter(WIDTHS,max_lrs_adam,s=12,marker='o', color=colors[iline],)
axes.text(2048,10**(slope * np.log10(2048) + intercept)*1.5,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
iline += 1


# GPT fixed
WIDTHS = [256,1024,4096]
x_log,y_log = np.log10(WIDTHS), np.log10(lrs_gpt_adam_fixed)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,label= 'fixed LN',color=colors[iline])
axes.scatter(WIDTHS,lrs_gpt_adam_fixed,s=12,marker='x', color=colors[iline],)
axes.text(512,10**(slope * np.log10(512) + intercept)*0.2,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')

x_log,y_log = np.log10(WIDTHS), np.log10(max_lrs_adam_fixed)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
y_log_pred = 10**(slope * (x_log) + intercept)
axes.plot(WIDTHS, y_log_pred,'--',color=colors[iline])
axes.scatter(WIDTHS,max_lrs_adam_fixed,s=12,marker='o', color=colors[iline],)
axes.text(1024,10**(slope * np.log10(1024) + intercept)*0.2,f'{np.round(slope,2)}',color=colors[iline], ha='center',fontsize=14, fontweight='bold')
iline += 1

axes.set_xscale('log')
axes.set_yscale('log')
# WIDTHS = [256,1024,4096,16384]
axes.set_xticks(WIDTHS)
axes.set_xticklabels(WIDTHS)
axes.set_xlabel('Width')
axes.set_ylabel('Learning rate')
#axes.set_title('ADAM')
#axes.set_ylim(0.0001,50)
plt.legend(frameon=True)#title='dataset, layers',loc='lower left')
plt.savefig(plot_path + f'mainfig_lrexponents_onlygpt_maxstablevsoptimal.png',dpi=300)
plt.clf()



# %%
