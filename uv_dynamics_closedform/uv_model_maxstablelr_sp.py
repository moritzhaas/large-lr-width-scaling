# %%

# finite SP, scaling widths:
# At which LR scaling does SP remain stable?
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from utils import mysave, myload, adjust_lightness

# optional: make plots look nice
import matplotlib as mpl
mpl.use('Agg')
plt.style.use('seaborn-whitegrid')
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
brightnesses= np.linspace(0.75,1/0.75,3)
colors=['tab:blue', 'tab:orange', 'tab:green'] # mup, sp, ntk
from utils import update, get_lr

seed = 0
np.random.seed(seed)  # For reproducibility

x, y = 1, 1

# because we use analytical update equations, we can scale to much larger widths! :)
width_values =[64,256,1024,4096,4*4096,16*4096,64*4096,256*4096] #16*4096=65536, 256*4096=1 048 576
eta_values = np.logspace(-5,1,30)
#eta_values = [0.01,0.1,0.4,0.49,0.5,0.95,1.0,1.7,1.9,2.0] # [1.0 / lambda_0]  # Example eta values for the three regimes
c_sp = 0 #without width scaling: blowup
phys_time = 20

indep_warmup = True
warmiter=200
#warmup = 0.1 #0.1 #None # fraction of steps
#decay = None #'cos' # None
#weight_decay = 0 #0.1

# Simulation parameters
num_iterations = 40


for setups in [(None,None,0), (None,0.1,0), ('cos',0.1,0), (None,None,0.1),('cos',None,0), (None,0.1,0.1), ('cos',0.1,0.1), (None,0.01,0), (None,0.2,0)]:
    decay, warmup, weight_decay = setups
    exp_name = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-{decay}{"-warmup-"+str(warmup) if not indep_warmup else "-warmiter"+str(warmiter)}-wd-{weight_decay}'
    if os.path.exists(f'stats/all_fs_lambdas' + exp_name + '.txt'):
        all_fs, all_lambdas, all_ts, eta_values, width_values = myload(f'stats/all_fs_lambdas' + exp_name + '.txt')
    else:
        np.random.seed(seed)
        all_fs, all_lambdas, all_ts = {}, {}, {}
        for width in width_values:
            for eta in eta_values:
                # train nets in SP
                max_eta = eta
                num_iterations = np.maximum(15, int(np.ceil(phys_time/eta)))#long enough for convergence
                save_iters = np.linspace(0,num_iterations,500,dtype=int) if num_iterations>500 else None
                if indep_warmup and warmiter is not None:
                    # ensure always same warmup iterations
                    warmup = warmiter / num_iterations

                u = np.random.randn(width)  # Example size of 10 for u
                v = np.random.randn(width)  # Example size of 10 for v

                # Initialize parameters for both methods
                f1 = x * np.dot(u, v)/width  # mup initialization -- multiplier version
                f2 = x * np.dot(u, v)/np.sqrt(width)  # ntk and sp initialization
                lambda_0 = x**2 * (np.linalg.norm(u)**2 + np.linalg.norm(v)**2) / width # This holds in both mup and ntp
                lambda_0_sp = x**2 * (np.linalg.norm(u)**2 + np.linalg.norm(v)**2 /width) / width # in sp, the v factor vanishes at init

                # SP
                f_t = f2
                lambda_t = lambda_0_sp
                ts = [0]
                f_values = [f_t]
                lambda_values = [lambda_t]

                for t in range(num_iterations):
                    eta = get_lr(max_eta, t, num_iterations, warmup=warmup, schedule=decay)
                    f_t, lambda_t = update(f_t, lambda_t, x, y, eta,param='sp',width=width,c_sp=c_sp,weight_decay= weight_decay)
                    if save_iters is None or t in save_iters:
                        ts.append(t+1)
                        f_values.append(f_t)
                        lambda_values.append(lambda_t)
                all_ts[(max_eta,width)] = ts
                all_fs[(max_eta,width)] = f_values
                all_lambdas[(max_eta,width)] = lambda_values

        mysave('stats/', f'all_fs_lambdas' + exp_name + '.txt', (all_fs, all_lambdas, all_ts, eta_values, width_values))

# %%
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

exp_name_plain = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-None'
exp_name_cos_warmup = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-cos-warmup-0.1'
exp_name_wd = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-None-wd-0.1'
exp_name_warmup = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.1'
exp_name_longwarmup = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.2-wd-0'
exp_name_shortwarmup = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.01-wd-0'
exp_name_cos = f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-cos-warmup-None'

exp_names_plain = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-None-wd-0' for seed in range(1)]  
exp_names_warmup = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.1-wd-0' for seed in range(1)]  
exp_names_wd = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-None-wd-0.1' for seed in range(1)]
exp_names_cos_warmup = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-cos-warmup-0.1-wd-0.1' for seed in range(1)]
exp_names_shortwarmup = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.01-wd-0' for seed in range(1)]  
exp_names_longwarmup = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.2-wd-0' for seed in range(1)]  

warmiter=50
# fixed amount of warmup iterations -> should eventually transition from n^{-1/2} to n^{-1}
exp_names_warmiter = [f'-sp-{c_sp}-warmiter{warmiter}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmup-0.1-wd-0' for seed in range(1)]
exp_names_longwarmiter = [f'-sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}-schedule-None-warmiter200-wd-0' for seed in range(1)]


# exps = {'plain SGD': ('tab:grey', exp_names_plain, 1.0),'SGD, wd 0.1': ('tab:orange', exp_names_wd, 1.0), 'SGD, warmup 0.1': ('tab:blue', exp_names_warmup, 0.5)}
# exps = {'plain SGD': ('tab:grey', exp_names_plain, 1.0),'SGD, warmup 0.01': (adjust_lightness('tab:blue',amount=brightnesses[::-1][0]), exp_names_shortwarmup, 0.5), 'SGD, warmup 0.1': (adjust_lightness('tab:blue',amount=brightnesses[::-1][1]), exp_names_warmup, 0.5), 'SGD, warmup 0.2': (adjust_lightness('tab:blue',amount=brightnesses[::-1][2]), exp_names_longwarmup, 0.5)}
exps = {'plain SGD': ('tab:grey', exp_names_plain, 1.0),f'SGD, {warmiter} warmupiter': (adjust_lightness('tab:blue',amount=brightnesses[::-1][0]), exp_names_warmiter, 1.0),f'SGD, {200} warmupiter': (adjust_lightness('tab:blue',amount=brightnesses[::-1][1]), exp_names_longwarmiter, 1.0), 'SGD, warmup 0.1': (adjust_lightness('tab:blue',amount=brightnesses[::-1][2]), exp_names_warmup, 0.5)}
# 'SGD, wd 0.1': ('tab:orange', [exp_name_wd], 1.0)
# 'SGD, cos, warmup 0.1': ('tab:green', exp_names_cos_warmup, 0.5)

def get_loglog_line(xs,ys):
    from scipy.stats import linregress
    xs = np.array(xs)
    ys = np.array(ys)
    valid_indices = ~np.isnan(xs) & ~np.isnan(ys)
    xs, ys = xs[valid_indices], ys[valid_indices]
    slope, intercept, r_value, p_value, std_err = linregress(np.log(xs), np.log(ys))
    return slope, intercept


plt.tight_layout()
fig, axes = plt.subplots(1, 1, figsize=(1*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
axes.set_xlabel('Width')
axes.set_ylabel(r'Largest stable learning rate')
fig_name = ''
for label in exps:
    color, exp_names, expon = exps[label]
    fig_name += '-' + label.replace(' ','').replace(',','').replace('(','').replace(')','').replace('.','')
    all_largest_stable_etas=[]
    for exp_name in exp_names:
        all_fs, all_lambdas, all_ts, eta_values, width_values = myload(f'stats/all_fs_lambdas' + exp_name + '.txt')

        largest_stable_etas=[]
        for iw, width in enumerate(width_values):
            final_fs = [all_fs[(eta,width)][-1] for eta in eta_values]
            try:
                largest_stable_etas.append(eta_values[np.where(np.abs(np.array(final_fs))<100)[0][-1]])
            except IndexError:
                largest_stable_etas.append(np.nan)
        all_largest_stable_etas.append(largest_stable_etas)


    all_largest_stable_etas = np.array(all_largest_stable_etas)
    largest_eta_mean = all_largest_stable_etas.mean(axis=0)
    lower, upper = np.quantile(all_largest_stable_etas,[0.025,0.975],axis=0)
    print(np.std(all_largest_stable_etas,axis=0))
    # plot naive training versus LR warmup and decay:
    axes.plot(width_values,largest_eta_mean, label=label, color=color,alpha=0.7)
    axes.fill_between(width_values, lower, upper, color=color, alpha=0.4)
    # axes.plot(width_values,largest_eta_mean[0]*width_values[0]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
    if 'plain' in label or '50 warmiter' in label:
        law_index = 4
    else:
        law_index = -1
    axes.plot(width_values,largest_eta_mean[law_index]*width_values[law_index]**expon *np.array(width_values)**(-expon),linestyle='--', color=color,alpha=0.7)
    # annotate the slope below the endpoint
    x_annotate = width_values[-1]
    y_annotate = largest_eta_mean[law_index]*width_values[law_index]**expon *width_values[-1]**(-expon)
    axes.annotate(f'-{expon}', xy=(0.95*x_annotate, y_annotate), xytext=(0.95*x_annotate, y_annotate * 0.9), fontsize=7, color=color,alpha=0.7,ha='center', va='top')
    
    #slope, intercept = get_loglog_line(width_values,largest_eta_mean)
    #axes.plot(width_values,np.exp(intercept) * width_values**slope,linestyle='--', color=color,alpha=0.7)
    # Annotate the slope
    #x_annotate = width_values[len(width_values) - 2]
    #y_annotate = np.exp(intercept) * x_annotate**slope
    #axes.annotate(f'{slope:.2f}', xy=(x_annotate, y_annotate), xytext=(x_annotate, y_annotate * 1.2), fontsize=8, color=color,alpha=0.7)

axes.set_yscale('log')
axes.set_xscale('log')
plt.legend()
plt.savefig(f'plots/final_stablelr_sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-seed-{seed}'+fig_name+'_exp.png')
plt.clf()

# %%










# %%
# plot f_t and lambda_t across widths
plt.tight_layout()
fig, axes = plt.subplots(1, 3, figsize=(3*onefigsize[0],1*onefigsize[1]))#plt.subplots(2, 4, figsize=(4*onefigsize[0],2*onefigsize[1]))
axes[0].set_title(r'Final function ($f_t$)')
axes[0].set_xlabel('Learning rate')
axes[0].set_ylabel(r'$f_t$')
axes[1].set_title(r'Lambda Dynamics ($\lambda_t$)')
axes[1].set_xlabel('Learning rate')
axes[1].set_ylabel(r'$\lambda_t$')
for iw, width in enumerate(width_values):
    final_fs = [all_fs[(eta,width)][-1] for eta in eta_values]
    final_lambdas = [all_lambdas[(eta,width)][-1] for eta in eta_values]
    axes[0].plot(eta_values, final_fs, label=width)
    axes[1].plot(eta_values,final_lambdas, label=width)

for idx in range(2):
    axes[idx].legend(title='width')
    axes[idx].set_xscale('log')

largest_stable_etas=[]
for iw, width in enumerate(width_values):
    final_fs = [all_fs[(eta,width)][-1] for eta in eta_values]
    largest_stable_etas.append(eta_values[np.where(np.abs(np.array(final_fs))<100)[0][-1]])

axes[2].set_title(r'Largest stable Learning Rate')
axes[2].set_xlabel('Width')
axes[2].set_ylabel(r'Learning rate')
axes[2].plot(width_values,largest_stable_etas)
axes[2].plot(width_values,np.array(width_values)**(-1.0),linestyle='--')
axes[2].set_yscale('log')
axes[2].set_xscale('log')

plt.savefig(f'plots/final_f_lambda_sp-{c_sp}-wid{width_values[0]}-{width_values[-1]}-eta-{eta_values[0]}-{eta_values[-1]}-time{phys_time}-x-{x}-y-{y}-schedule-{decay}-warmup-{warmup}.png')
plt.clf()
