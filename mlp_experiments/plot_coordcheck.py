# %%
"""
Plot the refined coordinate checks and other relevant internal statistics such as activation sparsity as a function of width for a fixed time step.
Requires pre-computed statistics from `coord_checks.py`.
"""

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

experiments = [('SGD', 0.0, 2), ('SGD',-0.5,2), ('SGD',0.0,1), ('LL-SGD',0.0,1), ('ADAM',-1.0,2), ('ADAM',0.0,2), ('ADAM',0.0,1), ('LL-ADAM',0.0,1)]
stat_names = ['delta_W_x_norm', 'activation_delta', 'delta_W_spectral_norms','activ_align_init_samept_alll', 'activ_align_init_otherpt_alll',  'eff_update_rank', 'activ0'] # 'activ_align_init_samept_alll', 'activ_align_init_otherpt_alll',  'eff_update_rank', 'activ0'

T = 10 # number of update steps
nepochs = 1
fine_widths = True  
extendedevaliter = 1
evaliter = 1
bs = 64
small_cifar_size = T * bs
finaleval = False  
#nhiddenlayers = 1
N_RUNS2 = N_RUNS = 4
#seed = 42
prefix = 'coord_check'
set_title = True
only_nomultipliers = True

all_files = find('stats*nomultcoord_checks_cifar10_SGD_lr0.0001_lrexp-0.5_allwidths_nlay2_seed42_*','./stats/cifar10/')

# (optim, nhiddenlayers, lr_expon)
exponents = {('SGD',2,-0.5): [-1,0,0.5],('SGD',2,0.0): [-1/2,1/2,1],('SGD',1,0.0): [-1/2,1],('LL-SGD',1,0.0): [0,1],
             ('ADAM',2,-1.0): [-1,0,0],('ADAM',2,0.0): [0,1,1],('ADAM',1,0.0): [0,1],('LL-ADAM',1,0.0): [0,1],
             ('ADAM',2,-0.5):[-1/2,1/2,1/2],('ADAM',1,-0.5): [-1/2,1/2],('LL-ADAM',1,-0.5): [0,1/2],}

llit_exponents = {('SGD',2,0.0): [0,1,1], ('SGD',1,0.0): [0,1],}

adjust_fontsize(3)
colors =sns.color_palette("rocket_r", n_colors=4)

def width_plot(WIDTHS, data, ylabel, filenam, num_rows = 1, titles = None, style = None, y_scale=None, colors = None, exponents = None, legend_title = None, fig_title = None, ylim = None):
    # data is dict where for each line key is label and value has shape width x plotidx x seeds (or if just one subplot: width x seeds)
    # ylabel and title can be lists of the length as number of subplots
    if num_rows>1: raise ValueError('Multiple rows not yet implemented.')
    for thiskey in data.keys():
        if len(data[thiskey].shape) == 2:
            num_plots = 1
            num_widths, num_seeds = data[thiskey].shape
        else:
            num_widths, num_plots, num_seeds = data[thiskey].shape
        break
    
    if num_plots % num_rows != 0: raise ValueError(f'Num plots: {num_plots}, but num rows: {num_rows}')
    num_lines = len(data.keys())
    num_cols = num_plots//num_rows
    
    if colors is None:
        colors = ['tab:blue','tab:green','tab:orange']
    plt.tight_layout()
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*onefigsize[0],num_rows*onefigsize[1]))
        
    for ikey,thiskey in enumerate(data.keys()):
        thisdata = data[thiskey]
        if style == 'relative':
            if ikey == 0: basedata = data[thiskey]
            minnumseeds = np.minimum(basedata.shape[-1],data[thiskey].shape[-1])
            thisdata = (data[thiskey][:,:,:minnumseeds]-basedata[:,:,:minnumseeds])/np.abs(basedata[:,:,:minnumseeds])

        if num_plots == 1:
            axes.plot(WIDTHS,thisdata.mean(axis=1),label=thiskey,color = colors[ikey])
            axes.fill_between(WIDTHS,np.quantile(thisdata,0.025,axis=1),np.quantile(thisdata,0.975,axis=1),alpha=0.4,color = colors[ikey])
        else:
            for iplot in range(num_plots):
                if num_rows == 1:
                    axes[iplot].plot(WIDTHS,thisdata[:,iplot,:].mean(axis=1),label=thiskey,color = colors[ikey])
                    axes[iplot].fill_between(WIDTHS,np.quantile(thisdata[:,iplot,:],0.025,axis=1),np.quantile(thisdata[:,iplot,:],0.975,axis=1),alpha=0.4,color = colors[ikey])
                    if exponents is not None:
                        axes[iplot].plot(WIDTHS,thisdata[-1,iplot,:].mean() * (np.array(WIDTHS)/WIDTHS[-1])**(exponents[iplot]),linestyle='--',color = colors[ikey])
                        if ikey == len(data.keys())-1 and exponents[iplot]!=0: #if thisdata[-1,iplot,:].mean() > thisdata[0,iplot,:].mean():
                            axes[iplot].text(WIDTHS[len(WIDTHS)//2], 2 * thisdata[-1,iplot,:].mean() * (np.array(WIDTHS[len(WIDTHS)//2])/WIDTHS[-1])**(exponents[iplot]),f'{np.round(exponents[iplot],3)}',color=colors[ikey], ha='center',fontsize=14,fontweight='bold')
                else:
                    axes[iplot//num_cols,iplot%num_cols].plot(WIDTHS,thisdata[:,iplot,:].mean(axis=1),label=thiskey,color = colors[ikey])
                    axes[iplot//num_cols,iplot%num_cols].fill_between(WIDTHS,np.quantile(thisdata[:,iplot,:],0.025,axis=1),np.quantile(thisdata[:,iplot,:],0.975,axis=1),alpha=0.4,color = colors[ikey])
                    if exponents is not None:
                        axes[iplot//num_cols,iplot%num_cols].plot(WIDTHS,thisdata[-1,iplot,:].mean() * (np.array(WIDTHS)/WIDTHS[-1])**(exponents[iplot]),linestyle='--',color = colors[ikey])
                        if ikey == len(data.keys())-1 and exponents[iplot]!=0: #if thisdata[-1,iplot,:].mean() > thisdata[0,iplot,:].mean():
                            axes[iplot//num_cols,iplot%num_cols].text(WIDTHS[len(WIDTHS)//2], (3 if exponents[iplot]!=0 else 2) * thisdata[-1,iplot,:].mean() * (np.array(WIDTHS[len(WIDTHS)//2])/WIDTHS[-1])**(exponents[iplot]),f'{np.round(exponents[iplot],3)}',color=colors[ikey], ha='center',alpha=0.99)
                
        
    #if isinstance(title,str):
    if num_plots == 1:
        axes.set_xlabel('Width')
        axes.set_xscale('log')
        axes.set_xticks(WIDTHS)
        axes.set_xticklabels(WIDTHS)
        axes.set_xticks([], minor=True)
        if isinstance(titles,list) and len(titles)==1: axes.set_title(titles[iplot])
        if isinstance(ylabel,list) and len(ylabel)==1: axes.set_ylabel(ylabel[iplot])
        if isinstance(ylabel,str): axes.set_ylabel(ylabel)
        if isinstance(ylim,list) and len(ylim)==2: axes.set_ylim(ylim[0],ylim[1])
        if y_scale is not None: axes.set_yscale(y_scale)
    else:
        if y_scale is not None:
            for ax in axes.flatten():
                ax.set_yscale(y_scale)
        for iplot in range(num_plots):
            if num_rows == 1:
                axes[iplot].set_xlabel('Width')
                axes[iplot].set_xscale('log')
                axes[iplot].set_xticks(WIDTHS)
                axes[iplot].set_xticklabels(WIDTHS)
                axes[iplot].set_xticks([], minor=True)
                if isinstance(titles,list) and len(titles)==num_plots: axes[iplot].set_title(titles[iplot])
                if isinstance(ylabel,list) and len(ylabel)==num_plots: axes[iplot].set_ylabel(ylabel[iplot])
                if isinstance(ylabel,str): axes[0].set_ylabel(ylabel)
                if isinstance(ylim,list) and len(ylim)==2: axes[0].set_ylim(ylim[0],ylim[1])
            else:
                axes[iplot//num_cols,iplot%num_cols].set_xlabel('Width')
                axes[iplot//num_cols,iplot%num_cols].set_xscale('log')
                axes[iplot//num_cols,iplot%num_cols].set_xticks(WIDTHS)
                axes[iplot//num_cols,iplot%num_cols].set_xticklabels(WIDTHS)
                axes[iplot//num_cols,iplot%num_cols].set_xticks([], minor=True)
                if isinstance(titles,list) and len(titles)==num_plots: axes[iplot//num_cols,iplot%num_cols].set_title(titles[iplot])
                if isinstance(ylabel,list) and len(ylabel)==num_plots: axes[iplot//num_cols,iplot%num_cols].set_ylabel(ylabel[iplot])
                if isinstance(ylim,list) and len(ylim)==2: axes[iplot//num_cols,iplot%num_cols].set_ylim(ylim[0],ylim[1])

    
    if num_rows>1 and isinstance(ylabel,list) and len(ylabel)==num_rows:
        for irow in range(num_rows):
            axes[irow,0].set_ylabel(ylabel[irow])
    if num_rows>1 and isinstance(titles, list) and len(titles) == num_plots//num_rows:
        for icol in range(num_plots//num_rows):
            axes[0,icol].set_title(titles[icol])
    
    axes[0].legend(title = legend_title,frameon=True)#,loc='lower left')#title='algo')
    if fig_title is not None: fig.suptitle(fig_title)
    plt.savefig(filenam,dpi=300)
    plt.clf()



for file in all_files:
    all_stats = myload(file)
    config = myload(file.replace('stats_','config_'))
    try:
        dataset, nhiddenlayers, OPTIM_ALGO, param, base_lr, lr_expon, WIDTHS, linear, seed, N_RUNS, llzeroinit, nomultipliers = config['dataset'], config['nhiddenlayers'], config['optim'], config['param'], config['lr'], config['lr_expon'], config['widths'], config['linear'], config['seed'], config['nruns'],config['llzeroinit'], config['nomultipliers']
    except KeyError:
        continue
        
    if only_nomultipliers and not nomultipliers: continue
    if 'SGD' in OPTIM_ALGO and base_lr not in [0.01,0.03,0.0001]: continue
    if 'ADAM' in OPTIM_ALGO and base_lr not in [0.0003, 1e-6]: continue
    seeds = [seed+irun for irun in range(N_RUNS)]

    rmsnormlayer = ('_rmsnorm_' in file)
    for stat_name in stat_names:
    #for (OPTIM_ALGO,lr_expon,nhiddenlayers) in experiments:
        EXP_NAME = f'{prefix}_{param}_{dataset}_{OPTIM_ALGO}_lr{base_lr}_lrexp{lr_expon}_allwidths_nlay{nhiddenlayers}{"_lin" if linear else ""}{"_llit" if llzeroinit else ""}{"_rmsnorm" if rmsnormlayer else ""}_seed{seed}_nruns{N_RUNS}'
        if stat_name in ['delta_W_x_norm', 'activation_delta', 'delta_W_spectral_norms'] and (param == 'sp' or rmsnormlayer):
            these_exponents = exponents[(OPTIM_ALGO,nhiddenlayers,lr_expon)] if (OPTIM_ALGO,nhiddenlayers,lr_expon) in exponents else None
            if llzeroinit and (OPTIM_ALGO,nhiddenlayers,lr_expon) in llit_exponents:
                these_exponents = llit_exponents[(OPTIM_ALGO,nhiddenlayers,lr_expon)]
        else:
            these_exponents = None

        if stat_name == 'delta_W_spectral_norms':
            y_label = r'$||\Delta W^l||_*/c^l$'
        elif stat_name == 'activation_delta':
            y_label = r'$||\Delta x^l||_{RMS}$'
        elif stat_name == 'delta_W_x_norm':
            y_label = r'$||\Delta W^l_t x^{l-1}_t||_{RMS}$'
        elif stat_name == 'activ_align_init':
            y_label = r'cos-sim($x^l_t$, $x^l_0$)'
        elif 'activ_align_init_samept' in stat_name:
            y_label = r'cos-sim($x^l_t$, $x^l_0$), same pt.'
        elif 'activ_align_init_otherpt' in stat_name:
            y_label = r'cos-sim($x^l_t$, $x^l_0$), other pt.'
        elif stat_name == 'eff_update_rank':
            y_label = r'$||\Delta W||_F / ||\Delta W||_*$'
        elif stat_name == 'activ0':
            y_label = r'fraction$(x^l==0)$'

        if stat_name == 'activ_align_init':
            titles = ['same training point', 'different point but same batch']
        else:
            titles = ['Input layer', 'Hidden layer', 'Output layer'] if nhiddenlayers > 1 else ['Input layer','Output layer']

        # seed x WIDTH x iter x layers
        iter_list, plot_list = [],[]
        for thisseed in seeds:
            sub_list = []
            for WIDTH in WIDTHS:
                for key in all_stats:
                    if key[0] == WIDTH and key[-1] == thisseed:
                        if stat_name == 'delta_W_spectral_norms':
                            iter_list.append([all_stats[key]['delta_W_spectral_norms'][ielem][0] for ielem in range(len(all_stats[key]['delta_W_spectral_norms']))])
                            sub_list.append([all_stats[key]['delta_W_spectral_norms'][ielem][1] for ielem in range(len(all_stats[key]['delta_W_spectral_norms']))])
                        elif stat_name == 'activation_delta':
                            iter_list.append([[all_stats[key][f'activation_delta_layer{i_l}'][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'activation_delta_layer{0}']))])
                            sub_list.append([[all_stats[key][f'activation_delta_layer{i_l}'][ielem][1] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'activation_delta_layer{0}']))])
                        elif stat_name == 'delta_W_x_norm':
                            iter_list.append([[all_stats[key][f'delta_W_x_norms'][ielem][0] for i_l in range(nhiddenlayers)] for ielem in range(len(all_stats[key][f'delta_W_x_norms']))]) #TODO: newer stats allow: nhiddenlayers+1
                            sub_list.append([[all_stats[key][f'delta_W_x_norms'][ielem][1][i_l] for i_l in range(nhiddenlayers)] for ielem in range(len(all_stats[key][f'delta_W_x_norms']))])
                        elif stat_name == 'activ_align_init':
                            iter_list.append([all_stats[key]['activ_align_init_samept'][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_samept']))])
                            sub_list.append([[all_stats[key]['activ_align_init_samept'][ielem][1],all_stats[key]['activ_align_init_otherpt'][ielem][1]] for ielem in range(len(all_stats[key]['activ_align_init_samept']))])
                        elif stat_name == 'activ_align_init_samept_alll':
                            iter_list.append([all_stats[key]['activ_align_init_samept_alll'][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_samept_alll']))])
                            sub_list.append([[all_stats[key]['activ_align_init_samept_alll'][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ_align_init_samept_alll']))])
                        elif stat_name == 'activ_align_init_otherpt_alll':
                            iter_list.append([all_stats[key]['activ_align_init_otherpt_alll'][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_otherpt']))])
                            sub_list.append([[all_stats[key]['activ_align_init_otherpt_alll'][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ_align_init_otherpt']))])
                        elif stat_name == 'activ0':
                            iter_list.append([all_stats[key][f'activ0_layer{i_l}'][ielem][0] for ielem in range(len(all_stats[key]['activ0_layer0']))])
                            sub_list.append([[all_stats[key][f'activ0_layer{i_l}'][ielem][1] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ0_layer0']))])
                        elif stat_name == 'eff_update_rank':
                            iter_list.append([all_stats[key]['delta_W_spectral_norms'][ielem][0] for ielem in range(len(all_stats[key]['delta_W_spectral_norms']))])
                            try:
                                sub_list.append([[all_stats[key]['delta_W_frob_norms'][ielem][1][i_l]/all_stats[key]['delta_W_spectral_norms'][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['delta_W_spectral_norms']))])
                            except ZeroDivisionError:
                                thisappend = [[np.nan for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['delta_W_spectral_norms']))]
                                for ielem in range(len(all_stats[key]['delta_W_spectral_norms'])):
                                    for i_l in range(nhiddenlayers+1):
                                        try:
                                            thisappend[ielem][i_l] = all_stats[key]['delta_W_frob_norms'][ielem][1][i_l]/all_stats[key]['delta_W_spectral_norms'][ielem][1][i_l]
                                        except ZeroDivisionError:
                                            thisappend[ielem][i_l] = np.nan
                                sub_list.append(thisappend)
                                    
                        if stat_name == 'delta_W_x_norm': # for older stats: last layer separately..
                            for it in range(len(sub_list[-1])):
                                sub_list[-1][it].append(all_stats[key][f'delta_W_x_norm'][it][1]) # last layer
            plot_list.append(sub_list)

        plot_array = np.array(plot_list)
        T_final = plot_array.shape[2]

        # reshape to width, layer, run
        label1 = T1 = 1
        data1 = plot_array[:,:,T1-1,:].transpose(1, 2, 0)
        #data1 = data1.reshape(data1.shape[0],data1.shape[1],-1)
        label2 = T2 = 2
        data2 = plot_array[:,:,T2-1,:].transpose(1, 2, 0)
        label3 = T3 = 4
        data3 = plot_array[:,:,T3-1,:].transpose(1, 2, 0)
        label4 = T4 = T_final
        data4 = plot_array[:,:,T_final-1,:].transpose(1, 2, 0)
        thisdict = {}
        
        for i_l in range(data1.shape[1]):
            if stat_name == 'delta_W_spectral_norms':
                if i_l == 0: width_expon = 1/2
                elif i_l == nhiddenlayers: width_expon = -1/2
                else: width_expon = 0
            elif stat_name == 'activation_delta':
                width_expon = 1/2 * (i_l < nhiddenlayers)
            elif stat_name == 'delta_W_x_norm':
                width_expon = 1/2 * (i_l < nhiddenlayers)
            else:
                width_expon = 0
            # width, layer, run
            for these_data in [data1,data2,data3,data4]:
                these_data[:,i_l,:] = np.array([[these_data[iw,i_l,irun]/(WIDTHS[iw]**width_expon) for irun in range(N_RUNS)] for iw,WIDTH in enumerate(WIDTHS)])
                if 'LL' in OPTIM_ALGO:
                    these_data = these_data[:,-1,:]
        for this_label, these_data in zip([label1,label2,label3,label4],[data1,data2,data3,data4]):
            thisdict[this_label] = these_data
        fig_title = f'{OPTIM_ALGO}, lr {base_lr}*n**{lr_expon}, {nhiddenlayers+1}-layer{", lin" if linear else ""}{", ll=0" if llzeroinit else ""}{", RMSNorm" if rmsnormlayer else ""}' if set_title else None
        
        this_plotpath = plot_path+f'{EXP_NAME.replace(" ","_").replace(".","")}/'
        if not os.path.exists(this_plotpath):
            os.makedirs(this_plotpath)
        width_plot(WIDTHS, thisdict, ylabel=y_label, titles=titles, filenam = this_plotpath+f'{stat_name}_mainfig_{"nomult_" if nomultipliers else ""}'+f'{EXP_NAME}_{"_notitle" if not set_title else ""}.png',y_scale='log' if stat_name not in ['activ_align_init','activ_align_init_alll','eff_update_rank'] else None,exponents=these_exponents, legend_title='update step',colors=colors,fig_title=None, ylim = [0,1] if stat_name == 'activ_align_init' else None)
        plt.clf()


# %%
# scaling law extrapolation
# at different scales, different terms dominate the training dynamics
# refined coordinate checks can be useful for predicting phase transitions as explained in Appendix E.1 of our paper https://arxiv.org/pdf/2505.22491

base_width, val, expon, val2, expon2 = 1024, 2e-2, -1/2, 9e-5, 0 #5e-2, -1/2, 5e-5, 0

# val * (width/base_width)**expon == val2 * (width/base_width2)**expon2
# assuming same base_width:
# (width/base_width)**(expon-expon2) == val2/val
# width = base_width * (val2/val1)**(1/(expon-expon2))

def crossing(base_width, vals, expons):
    return base_width * (vals[1]/vals[0])**(1/(expons[0]-expons[1]))

crossing(base_width, [val,val2], [expon,expon2])

# %%