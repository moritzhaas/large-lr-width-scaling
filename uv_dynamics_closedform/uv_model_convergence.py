# %%
# how many steps is limit accurately approximated by width?
# plot: width vs max T: $|f_t-\mathring{f}_t|<0.001$ for all $t\le T$.

# also for different fixed t: plot width vs |f_t-\mathring{f}_t| for all params
# but convergence to f=y is a confounder -> sample new random points (x,y) from N(0,I)?

import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from utils import mysave, myload, adjust_lightness, get_lr, update

# optional: make plots look nice
import matplotlib as mpl
mpl.use('Agg')
plt.style.use('seaborn-whitegrid')
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
brightnesses= np.linspace(0.5,1.75,4)
colors=['tab:blue', 'tab:orange', 'tab:green'] # mup, sp, ntk

num_seeds = 100
seed = 0
np.random.seed(seed)  # For reproducibility

x, y = np.random.randn(1),np.random.randn(1) #1, 1

param = 'mup' # 'mup', 'ntp', 'sp'
width_values =[64,256,1024,4096,4*4096,16*4096]
max_eta = eta = 0.001 #np.logspace(-5,1,30)# if param=='sp' else np.logspace(-3,3,30)
#eta_values = [0.01,0.1,0.4,0.49,0.5,0.95,1.0,1.7,1.9,2.0] # [1.0 / lambda_0]  # Example eta values for the three regimes
c_sp = 1
phys_time = 20
mup_ll0 = False
differing_initialf = True

#decay = 'cos' #'cos' # None
#warmup = 0.1 #0.1 #None # fraction of steps
#weight_decay = 0 #0.1

num_iterations = 40

for seed in range(num_seeds):
    for setups in [(None,None,0), ('cos',None,0), (None,0.1,0), ('cos',0.1,0), (None,None,0.1), (None,0.1,0.1), ('cos',0.1,0.1)]:
        decay, warmup, weight_decay = setups
        for param in ['mup','sp','ntp']:
            exp_name = f'-{param}{"-ll0" if mup_ll0 and param=="mup" else ""}{"-differingf" if differing_initialf else ""}-wid{width_values[0]}-{width_values[-1]}-eta-{max_eta}-time{phys_time}-normalxy-numiter{num_iterations}-seed-{seed}-schedule-{decay}-warmup-{warmup}-wd-{weight_decay}'

            if os.path.exists(f'stats/fvslim/fvslim_' + exp_name + '.txt'):
                all_fs, all_lambdas, all_ts, eta_values, width_values = myload(f'stats/fvslim/fvslim_' + exp_name + '.txt')
                # with open(f'stats/fvslim/all_fs_lambdas' +exp_name + '.pkl', 'rb') as f:
                #     all_fs, all_lambdas, eta_values, width_values = pickle.load(f)
            else:
                all_fs, all_lambdas, all_ts,all_diffs = {}, {}, {}, {}
                for width in width_values:
                    np.random.seed(seed)  # For reproducibility
                    # train nets in SP
                    #num_iterations = np.maximum(15, int(np.ceil(phys_time/max_eta)))#long enough for convergence
                    #save_iters = np.linspace(0,num_iterations,500,dtype=int) if num_iterations>500 else None

                    u = np.random.randn(width)
                    v = np.random.randn(width) 

                    # Initialize parameters for both methods
                    f1 = x * np.dot(u, v)/width  # mup initialization -- multiplier version
                    f2 = x * np.dot(u, v)/np.sqrt(width)  # ntk and sp initialization
                    lambda_0 = x**2 * (np.linalg.norm(u)**2 + np.linalg.norm(v)**2) / width # This holds in both mup and ntp
                    lambda_0_mupll0 = x**2 * (np.linalg.norm(u)**2) / width # when initializing last layer weights to 0
                    lambda_0_sp = x**2 * (np.linalg.norm(u)**2 + np.linalg.norm(v)**2 /width) / width # in sp, the v factor vanishes at init

                    f_t = 0 if param == 'mup' else f2 # for fairness start all from same initial condition. instead of: f1 if param == 'mup' else f2
                    lambda_t = lambda_0_sp if param == 'sp' else lambda_0
                    f_t_lim = 0 if param == 'mup' else f2 #TODO: this should be from N(0,1), not from finite width! But we want to start from same condition and still Var=1..
                    lambda_t_lim = 1 * x**2 if param == 'sp' else 2 * x**2
                    if differing_initialf:
                        f_t = f1 if param == 'mup' else f2
                        f_t_lim = 0 if param == 'mup' else x * np.random.randn(1)

                    if param == 'mup' and mup_ll0:
                        lambda_t = lambda_0_mupll0
                        lambda_t_lim = x**2
                    ts = [0]
                    f_values = [f_t]
                    lambda_values = [lambda_t]
                    diff_to_lims = [np.abs(f_t_lim-f_t)]
                    print(param, setups, width, np.abs(f_t_lim-f_t), np.abs(lambda_t_lim-lambda_t))

                    for t in range(num_iterations):
                        eta = get_lr(max_eta, t, num_iterations, warmup=warmup, schedule=decay)
                        f_t, lambda_t = update(f_t, lambda_t, x, y, eta,param=param,width=width,c_sp=c_sp,weight_decay=weight_decay)
                        if param == 'mup':
                            f_t_lim, lambda_t_lim = update(f_t_lim, lambda_t_lim, x, y, eta,param='mup',width=width,weight_decay=weight_decay)
                        elif param == 'sp':
                            try:
                                chi = f_t_lim - y
                                if c_sp == 1:
                                    f_t_lim = (1-weight_decay*eta)**2 * f_t_lim - eta * (1-weight_decay*eta) * chi * lambda_t_lim
                                    # lambda_t remains constant in infinite width limit without weight decay
                                    lambda_t_lim =(1-weight_decay*eta) * lambda_t_lim
                                else: raise NotImplementedError()
                            except OverflowError:
                                f_t_lim, lambda_t_lim = np.nan, np.nan
                        elif param == 'ntp':
                            try:
                                chi = f_t_lim - y
                                f_t_lim = (1-weight_decay*eta)**2*f_t_lim - eta * (1-weight_decay*eta) * chi * lambda_t_lim
                                # lambda_t remains constant in infinite width limit without weight decay
                                lambda_t_lim =(1-weight_decay*eta) * lambda_t_lim
                            except OverflowError:
                                f_t_lim, lambda_t_lim = np.nan, np.nan
                        x, y = np.random.randn(1),np.random.randn(1) # new data without signal for the next iteration

                        # if save_iters is None or t in save_iters:
                        ts.append(t+1)
                        f_values.append(f_t)
                        lambda_values.append(lambda_t)
                        diff_to_lims.append(np.abs(f_t_lim-f_t))
                    all_ts[(max_eta,width)] = ts
                    all_fs[(max_eta,width)] = f_values
                    all_lambdas[(max_eta,width)] = lambda_values
                    all_diffs[(max_eta,width)] = diff_to_lims

                mysave('stats/fvslim/', f'fvslim_' + exp_name + '.txt', (all_fs, all_lambdas, all_ts, all_diffs, width_values))


# %%
max_eta = 0.001
#mup_ll0 = False
# plot for different fixed t: width vs |f_t-\mathring{f}_t| for all params
exp_names_plain = [f'-wid{width_values[0]}-{width_values[-1]}-eta-{max_eta}-time{phys_time}-normalxy-numiter40-seed-{seed}-schedule-None-warmup-None-wd-0' for seed in range(num_seeds)]
exp_names_wd = [f'-wid{width_values[0]}-{width_values[-1]}-eta-{max_eta}-time{phys_time}-normalxy-numiter40-seed-{seed}-schedule-None-warmup-None-wd-0.1' for seed in range(num_seeds)]
exp_names_warmup = [f'-wid{width_values[0]}-{width_values[-1]}-eta-{max_eta}-time{phys_time}-normalxy-numiter40-seed-{seed}-schedule-None-warmup-0.1-wd-0' for seed in range(num_seeds)]
exp_names_cos_warmup = [f'-wid{width_values[0]}-{width_values[-1]}-eta-{max_eta}-time{phys_time}-normalxy-numiter40-seed-{seed}-schedule-cos-warmup-0.1-wd-0' for seed in range(num_seeds)]
exps = {'plain SGD': ('tab:grey', exp_names_plain, 0.5),'SGD, wd 0.1': ('tab:orange', exp_names_wd, 0.5), 'SGD, warmup 0.1': ('tab:green', exp_names_warmup, 0.5)}
# 'SGD, cos, warmup 0.1': ('tab:blue', exp_names_cos_warmup, 0.5),
# 'SGD, wd 0.1': ('tab:orange', exp_names_wd, 0.5), 
# 'plain SGD': ('tab:grey', exp_names_plain, 0.5),

def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    expon = (np.log(y[0])-np.log(y[1]))/(np.log(x[0])-np.log(x[1]))
    if type(expon)==np.ndarray: expon = expon[0]
    return expon

if not os.path.exists('plots/difftolim/'):
    os.makedirs('plots/difftolim/')

all_diffs_lim_mean, all_diffs_std, exponents = {}, {}, {}
#T=20 #1,2,5,10,20
for T in [2,5,10,20]:
    plt.tight_layout()
    fig, axes = plt.subplots(1, 1, figsize=(1*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
    axes.set_xlabel('Width')
    axes.set_ylabel(r'$|f_t-\mathring{f}_t|$')
    fig_name = ''
    ymin,ymax = 10, 0
    for label in exps:
        color, exp_names, expon = exps[label]
        fig_name += '-' + label.replace(' ','').replace(',','').replace('(','').replace(')','').replace('.','')
        all_diffs_lim={param: [] for param in ['mup','sp','ntp']}
        for linestyle, param in zip(['-','--',':'], ['mup','sp','ntp']):
            for exp_name in exp_names:
                all_fs, all_lambdas, all_ts, all_diffs, width_values = myload(f'stats/fvslim/fvslim_-'+ param +("-ll0" if mup_ll0 and param == 'mup' else "")+("-differingf" if differing_initialf else "")+ exp_name + '.txt')
                for eta,width in all_ts: break

                diffs=[]
                for iw, width in enumerate(width_values):
                    diffs.append(all_diffs[(eta,width)][np.where(np.abs(np.array(all_ts[(eta,width)])-T)<0.5)[0][-1]])
                all_diffs_lim[param].append(diffs)

            all_diffs_lim[param] = np.array(all_diffs_lim[param])
            all_diffs_lim_mean[(param, label, T)] = all_diffs_lim[param].mean(axis=0)
            lower, upper = np.quantile(all_diffs_lim[param],[0.025,0.975],axis=0)
            lower, upper = lower.flatten(), upper.flatten()
            all_diffs_std[(param, label, T)] = np.std(all_diffs_lim[param],axis=0)
            exponents[(param, label, T)] = exponent((width_values[0],width_values[-1]),(all_diffs_lim_mean[(param, label, T)][0],all_diffs_lim_mean[(param, label, T)][-1]))
            #print(param, label, T, exponents[(param, label, T)])
            # plot naive training versus LR warmup and decay:
            axes.plot(width_values,all_diffs_lim_mean[(param, label, T)], label=label if param=='mup' else None, linestyle=linestyle, color=color,alpha=0.7)
            axes.fill_between(width_values, lower, upper, color=color, alpha=0.18)
            if np.max(all_diffs_lim_mean[(param, label, T)])>ymax:
                ymax = np.max(all_diffs_lim_mean[(param, label, T)])
            if np.min(all_diffs_lim_mean[(param, label, T)])<ymin:
                ymin = np.min(all_diffs_lim_mean[(param, label, T)])
            #axes.plot(width_values,all_diffs_lim_mean[0]*width_values[0]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
    axes.set_title(f'T={T}, eta={eta}, mup=-, sp=- -, ntp=:')
    axes.set_ylim(ymin/5,ymax*5)
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt.legend()
    plt.savefig(f'plots/difftolim/difftolim_T{T}-wid{width_values[0]}-{width_values[-1]}-eta-{eta}-time{phys_time}-seed-{seed}'+fig_name+f'{"-ll0" if mup_ll0 else ""}{"-differingf" if differing_initialf else ""}.png')
    plt.clf()

# std of diff to lim: large -> closeness to lim depends very much on random seed
for T in [2,5,10,20]:
    plt.tight_layout()
    fig, axes = plt.subplots(1, 1, figsize=(1*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
    axes.set_xlabel('Width')
    axes.set_ylabel(r'$Std(|f_t-\mathring{f}_t|)$')
    fig_name = ''
    ymin,ymax = 10, 0
    for label in exps:
        color, exp_names, expon = exps[label]
        fig_name += '-' + label.replace(' ','').replace(',','').replace('(','').replace(')','').replace('.','')
        for linestyle, param in zip(['-','--',':'], ['mup','sp','ntp']):
            for exp_name in exp_names:
                all_fs, all_lambdas, all_ts, all_diffs, width_values = myload(f'stats/fvslim/fvslim_-'+ param + ("-ll0" if mup_ll0 and param == 'mup' else "")+("-differingf" if differing_initialf else "")+ exp_name + '.txt')
                for eta,width in all_ts: break
            # all_diffs_std[(param, label, T)] # = np.std(all_diffs_lim[param],axis=0)
            
            axes.plot(width_values,all_diffs_std[(param, label, T)], label=label if param=='mup' else None, linestyle=linestyle, color=color,alpha=0.7)
            if np.max(all_diffs_std[(param, label, T)])>ymax:
                ymax = np.max(all_diffs_std[(param, label, T)])
            if np.min(all_diffs_std[(param, label, T)])<ymin:
                ymin = np.min(all_diffs_std[(param, label, T)])
            #axes.plot(width_values,all_diffs_lim_mean[0]*width_values[0]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
    axes.set_title(f'T={T}, eta={eta}, mup=-, sp=- -, ntp=:')
    axes.set_ylim(ymin/5,ymax*5)
    axes.set_yscale('log')
    axes.set_xscale('log')
    plt.legend()
    plt.savefig(f'plots/difftolim/std_difftolim_T{T}-wid{width_values[0]}-{width_values[-1]}-eta-{eta}-time{phys_time}-seed-{seed}'+fig_name+f'{"-ll0" if mup_ll0 else ""}{"-differingf" if differing_initialf else ""}.png')
    plt.clf()

for T in [2,5,10,20]:
    plt.tight_layout()
    fig, axes = plt.subplots(1, 1, figsize=(1*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
    axes.set_xlabel('Width')
    axes.set_ylabel(r'$Std(|f_t-\mathring{f}_t|)/Mean(|f_t-\mathring{f}_t|)$')
    fig_name = ''
    ymin,ymax = 10, 0
    for label in exps:
        color, exp_names, expon = exps[label]
        fig_name += '-' + label.replace(' ','').replace(',','').replace('(','').replace(')','').replace('.','')
        for linestyle, param in zip(['-','--',':'], ['mup','sp','ntp']):
            for exp_name in exp_names:
                all_fs, all_lambdas, all_ts, all_diffs, width_values = myload(f'stats/fvslim/fvslim_-'+ param + ("-ll0" if mup_ll0 and param == 'mup' else "")+("-differingf" if differing_initialf else "")+ exp_name + f'.txt')
                for eta,width in all_ts: break
            # all_diffs_std[(param, label, T)] # = np.std(all_diffs_lim[param],axis=0)
            
            std_over_mean = all_diffs_std[(param, label, T)].flatten()/all_diffs_lim_mean[(param, label, T)].flatten()
            axes.plot(width_values,std_over_mean, label=label if param=='mup' else None, linestyle=linestyle, color=color,alpha=0.7)
            # if np.max(std_over_mean)>ymax:
            #     ymax = np.max(std_over_mean)
            # if np.min(std_over_mean)<ymin:
            #     ymin = np.min(std_over_mean)
            #axes.plot(width_values,all_diffs_lim_mean[0]*width_values[0]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
    axes.set_title(f'T={T}, eta={eta}, mup=-, sp=- -, ntp=:')
    #axes.set_ylim(ymin/5,ymax*5)
    #axes.set_yscale('log')
    axes.set_xscale('log')
    plt.legend()
    plt.savefig(f'plots/difftolim/std_over_mean_difftolim_T{T}-wid{width_values[0]}-{width_values[-1]}-eta-{eta}-time{phys_time}-seed-{seed}'+fig_name+f'{"-ll0" if mup_ll0 else ""}{"-differingf" if differing_initialf else ""}.png')
    plt.clf()


Ts = np.arange(num_iterations)#[2,5,10,20]
for T in Ts:
    for label in exps:
        color, exp_names, expon = exps[label]
        all_diffs_lim={param: [] for param in ['mup','sp','ntp']}
        for param in ['mup','sp','ntp']:
            for exp_name in exp_names:
                all_fs, all_lambdas, all_ts, all_diffs, width_values = myload(f'stats/fvslim/fvslim_-'+ param +("-ll0" if mup_ll0 and param == 'mup' else "")+("-differingf" if differing_initialf else "")+ exp_name + f'.txt')
                for eta,width in all_ts: break

                diffs=[]
                for iw, width in enumerate(width_values):
                    diffs.append(all_diffs[(eta,width)][np.where(np.abs(np.array(all_ts[(eta,width)])-T)<0.5)[0][-1]])
                all_diffs_lim[param].append(diffs)

            all_diffs_lim[param] = np.array(all_diffs_lim[param])
            all_diffs_lim_mean[(param, label, T)] = all_diffs_lim[param].mean(axis=0)
            exponents[(param, label, T)] = exponent((width_values[0],width_values[-1]),(all_diffs_lim_mean[(param, label, T)][0],all_diffs_lim_mean[(param, label, T)][-1]))

plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(1*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
axes.set_xlabel('T')
axes.set_ylabel(r'Exponent$(|f_t-\mathring{f}_t|)$')
fig_name = ''
ymin,ymax = 10, 0
for label in exps:
    color, exp_names, expon = exps[label]
    fig_name += '-' + label.replace(' ','').replace(',','').replace('(','').replace(')','').replace('.','')
    for linestyle, param in zip(['-','--',':'], ['mup','sp','ntp']):
        these_exp = [exponents[(param, label, T)] for T in Ts]
        axes.plot(Ts,these_exp, label=label if param=='mup' else None, linestyle=linestyle, color=color,alpha=0.7)
        # if np.max(these_exp)>ymax:
        #     ymax = np.max(these_exp)
        # if np.min(all_diffs_std[(param, label, T)])<ymin:
        #     ymin = np.min(all_diffs_std[(param, label, T)])
        #axes.plot(width_values,all_diffs_lim_mean[0]*width_values[0]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
axes.set_title(f'eta={eta}, mup=-, sp=- -, ntp=:')
# axes.set_ylim(ymin/5,ymax*5)
# axes.set_yscale('log')
# axes.set_xscale('log')
plt.legend()
plt.savefig(f'plots/difftolim/exponents_difftolim_T{Ts[0]}_{Ts[1]}-wid{width_values[0]}-{width_values[-1]}-eta-{eta}-time{phys_time}-seed-{seed}'+fig_name+f'{"-ll0" if mup_ll0 else ""}{"-differingf" if differing_initialf else ""}.png')
plt.clf()

# %%