# %%
"""
Plot the refined coordinate checks and other relevant internal statistics such as activation sparsity as a function of width for a fixed time step.
Requires pre-computed statistics from `main_mlp_allwidths.py` run with extended eval.
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
from utils.plot_utils import adjust_lightness, width_plot
from utils.eval import exponent, scaling_law
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']
brightnesses= np.linspace(0.5,1.75,4)
plot_path = './figures/coord_checks/' 
if  not os.path.exists(plot_path):
    os.makedirs(plot_path)
mpl.rcParams['text.usetex'] = True  # Use system LaTeX for rendering

experiments = [('SGD', 0.0, 2, 'sp'),('SGD', 0.0, 2, 'mup'), ('ADAM',0.0,2, 'sp'), ('ADAM',0.0,2, 'mup'), ('ADAM',-1.0,2, 'sp'), ('SGD',0,5, 'sp'), ('SGD',0.0,7, 'sp'), ('ADAM',0.0,7, 'sp')]
stat_names = ['delta_W_x_law', 'W_delta_x_law','delta_W_x_norms', 'W_delta_x_norms', 'activation_delta', 'delta_W_spectral_norms','activ_align_init_samept_alll', 'activ_align_init_otherpt_alll',  'eff_update_rank', 'activ0'] # 'activ_align_init_samept_alll', 'activ_align_init_otherpt_alll',  'eff_update_rank', 'activ0'

T = 10
nepochs = 1
fine_widths = True  
extendedevaliter = 1
evaliter = 1
bs = 64
small_cifar_size = T * bs
finaleval = False  
N_RUNS2 = N_RUNS = 4
prefix = 'rcc'
set_title = True
only_nomultipliers = True
seeds = np.arange(421,425) # np.arange(421,425) for Delta, 521-524 for delta


# (optim, nhiddenlayers, lr_expon)
exponents = {('SGD',2,-0.5): [-1,0,0.5],('SGD',2,0.0): [-1/2,1/2,1],('SGD',1,0.0): [-1/2,1],('LL-SGD',1,0.0): [0,1],
             ('ADAM',2,-1.0): [-1,0,0],('ADAM',2,0.0): [0,1,1],('ADAM',1,0.0): [0,1],('LL-ADAM',1,0.0): [0,1],
             ('ADAM',2,-0.5):[-1/2,1/2,1/2],('ADAM',1,-0.5): [-1/2,1/2],('LL-ADAM',1,-0.5): [0,1/2],}

llit_exponents = {('SGD',2,0.0): [0,1,1], ('SGD',1,0.0): [0,1],}


def load_multiseed_stats(filenames):
    all_stats,all_configs={},{}
    if os.path.exists(filenames[0].replace('stats_','config_')):
        for filename in filenames:
            try:
                temp_stats = myload(filename)
                if isinstance(temp_stats, list):
                    tag_dict, temp_stats = temp_stats
            except RuntimeError: continue
            
            for key in temp_stats.keys():
                #if 'cifar10' in filename and len(temp_stats[key]['epoch']) < 21: continue
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

# for every experiment, gather stats and plot
for experiment in experiments:
    optim_algo, lr_expon, nlay, param = experiment

    filenames=[]
    for seed in seeds:
        filter_string = f'stats_mlp*nlay{nlay}*seed{seed}*'
        for filenam in find(filter_string, './stats/cifar10/'):
            config = myload(filenam.replace('stats_','config_'))
            if config.optim == optim_algo and config.param == param and config.nhiddenlayers == nlay and config.lrexp == lr_expon and config.linear == False and config.seed == seed and config.width_choice=='standard' and config.extendedevaliter==20 and config.llzeroinit == False:
                filenames.append(filenam)
                dataset, nhiddenlayers, OPTIM_ALGO, param, base_lr, lr_expon, width_choice, linear, seed, llzeroinit, nomultipliers = config.dataset, config.nhiddenlayers, config.optim, config.param, config.lr, config.lrexp, config.width_choice, config.linear, config.seed,config.llzeroinit, config.nomultipliers
    if len(filenames) == 0:
        print(f'No files found for {experiment},  '+filter_string)
        continue
    
    all_stats, hps = load_multiseed_stats(filenames)
    WIDTHS, LRS, RHOS=[],[],[]
    for key in sorted(all_stats, key=lambda key: int(key[0])):
        if key[0] not in WIDTHS: WIDTHS.append(key[0])
        if key[1] not in LRS: LRS.append(key[1])
        if key[2] not in RHOS: RHOS.append(key[2])
            
    WIDTHS, LRS, RHOS = np.sort(WIDTHS),np.sort(LRS),np.sort(RHOS)

    if only_nomultipliers and not nomultipliers: continue
    if 'SGD' in OPTIM_ALGO and base_lr not in [0.032,0.1]: continue
    if 'ADAM' in OPTIM_ALGO and base_lr not in [0.00032, 0.1]: continue
    N_RUNS = 4

    rmsnormlayer = False #('_rmsnorm_' in file)
    for stat_name in stat_names:
        EXP_NAME = f'{prefix}_{param}_{dataset}_{OPTIM_ALGO}_lr{base_lr}_lrexp{lr_expon}_allwidths_nlay{nhiddenlayers}{"_lin" if linear else ""}{"_llit" if llzeroinit else ""}{"_rmsnorm" if rmsnormlayer else ""}_seed{seed}_nruns{N_RUNS}'
        if stat_name in ['delta_W_x_norms','W_delta_x_norms', 'activation_delta', 'delta_W_spectral_norms'] and (param == 'sp' or rmsnormlayer):
            these_exponents = exponents[(OPTIM_ALGO,nhiddenlayers,lr_expon)] if (OPTIM_ALGO,nhiddenlayers,lr_expon) in exponents else None
            if llzeroinit and (OPTIM_ALGO,nhiddenlayers,lr_expon) in llit_exponents:
                these_exponents = llit_exponents[(OPTIM_ALGO,nhiddenlayers,lr_expon)]
        else:
            these_exponents = None

        if stat_name == 'delta_W_spectral_norms':
            y_label = r'$||\Delta W_t^l||_*/c^l$'
        elif stat_name == 'activation_delta':
            y_label = r'$||\Delta x_t^l||_{RMS}$'
        elif stat_name == 'delta_W_x_norms':
            y_label = r'$||\Delta W_t^l x_t^{l-1}||_{RMS}$'
        elif stat_name == 'W_delta_x_norms':
            y_label = r'$||W_0^l \Delta x_t^{l-1}||_{RMS}$'
        elif stat_name == 'delta_W_x_law':
            y_label = r'$||\Delta W_t^l x_t^{l-1}||/(||\Delta W_t^l|| ||x_t^{l-1}||)$'
        elif stat_name == 'W_delta_x_law':
            y_label = r'$||W_0^l \Delta x_t^{l-1}||/(||W_0^l|| ||\Delta x_t^{l-1}||)$'
        elif stat_name == 'activ_align_init':
            y_label = r'cos-sim($x^l_t$, $x^l_0$)'
        elif 'activ_align_init_samept' in stat_name:
            y_label = r'cos-sim($x^l_t$, $x^l_0$), same pt.'
        elif 'activ_align_init_otherpt' in stat_name:
            y_label = r'cos-sim($x^l_t$, $x^l_0$), other pt.'
        elif stat_name == 'eff_update_rank':
            y_label = r'$||\Delta W_t||_F / ||\Delta W_t||_*$'
        elif stat_name == 'activ0':
            y_label = r'fraction$(x_t^l==0)$'

        if stat_name == 'activ_align_init':
            titles = ['same training point', 'different point but same batch']
        else:
            titles = ['Input layer', 'Hidden layer', 'Output layer'] if nhiddenlayers > 1 else ['Input layer','Output layer']

        # seed x WIDTH x iter x layers
        iter_list, plot_list = [],[]
        N_RUNS=4
        for iseed,thisseed in enumerate(seeds):
            sub_list = []
            for WIDTH in WIDTHS:
                for key in all_stats:
                    if len(all_stats[key]['delta_W_spectral_norms'])>4:
                        raise ValueError(f'Collected too many stats {len(all_stats[key]["delta_W_spectral_norms"])} for {key}')
                    if key[0] == WIDTH:
                        N_RUNS = np.minimum(N_RUNS,len(all_stats[key]['delta_W_spectral_norms']))
                        if iseed >= N_RUNS: continue
                        if stat_name == 'delta_W_spectral_norms':
                            iter_list.append([all_stats[key]['delta_W_spectral_norms'][iseed][ielem][0] for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0]))])
                            sub_list.append([all_stats[key]['delta_W_spectral_norms'][iseed][ielem][1] for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0]))])
                        elif stat_name == 'activation_delta':
                            iter_list.append([[all_stats[key][f'activation_delta_layer{i_l}'][iseed][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'activation_delta_layer{0}'][0]))])
                            sub_list.append([[all_stats[key][f'activation_delta_layer{i_l}'][iseed][ielem][1] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'activation_delta_layer{0}'][0]))])
                        elif stat_name == 'delta_W_x_norms':
                            iter_list.append([[all_stats[key][f'delta_W_x_norms'][iseed][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'delta_W_x_norms'][0]))])
                            sub_list.append([[all_stats[key][f'delta_W_x_norms'][iseed][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'delta_W_x_norms'][0]))])
                        elif stat_name == 'W_delta_x_norms':
                            iter_list.append([[all_stats[key][f'W_delta_x_norms'][iseed][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'W_delta_x_norms'][0]))])
                            sub_list.append([[all_stats[key][f'W_delta_x_norms'][iseed][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'W_delta_x_norms'][0]))])
                        elif stat_name == 'delta_W_x_law':
                            # fanin*2-norm/(Frobnorm*2-norm)
                            # sub_list is iter x layers given (seed,width)
                            iter_list.append([[all_stats[key][f'delta_W_x_norms'][iseed][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'delta_W_x_norms'][0]))])
                            if dataset == 'cifar10':
                                fan_ins = [32 * 32 * 3, WIDTH, WIDTH]
                            subsub_list = [[np.nan for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['delta_W_x_norms'][0]))]
                            for ielem in range(len(all_stats[key]['delta_W_x_norms'][0])):
                                for i_l in range(nhiddenlayers+1):
                                    try:
                                        subsub_list[ielem][i_l] = fan_ins[i_l] * all_stats[key]['delta_W_x_norms'][iseed][ielem][1][i_l] / (all_stats[key]['delta_W_frob_norms'][iseed][ielem][1][i_l]*all_stats[key][f'activation_l2_layer{i_l}'][iseed][ielem][1])
                                    except ZeroDivisionError:
                                        print(f'deltaWx law failed {i_l}')
                                        subsub_list[ielem][i_l] = np.nan
                            sub_list.append(subsub_list)
                        elif stat_name == 'W_delta_x_law':
                            # fanin*2-norm/(Frobnorm*2-norm)
                            # sub_list is iter x layers given (seed,width)
                            iter_list.append([[all_stats[key][f'delta_W_x_norms'][iseed][ielem][0] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key][f'delta_W_x_norms'][0]))])
                            if dataset == 'cifar10':
                                fan_ins = [32 * 32 * 3, WIDTH, WIDTH]
                            subsub_list = [[np.nan for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['W_delta_x_norms'][0]))]
                            for ielem in range(len(all_stats[key]['W_delta_x_norms'][0])):
                                for i_l in range(nhiddenlayers+1):
                                    try:
                                        subsub_list[ielem][i_l] = fan_ins[i_l] * all_stats[key]['W_delta_x_norms'][iseed][ielem][1][i_l] / (all_stats[key]['W_init_frob_norm'][iseed][0][i_l]*all_stats[key][f'activation_delta_layer{i_l}'][iseed][ielem][1])
                                    except ZeroDivisionError:
                                        print(f'Wdeltax law failed {i_l}')
                                        subsub_list[ielem][i_l] = np.nan
                            sub_list.append(subsub_list)
                        elif stat_name == 'activ_align_init':
                            iter_list.append([all_stats[key]['activ_align_init_samept'][iseed][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_samept'][0]))])
                            sub_list.append([[all_stats[key]['activ_align_init_samept'][iseed][ielem][1],all_stats[key]['activ_align_init_otherpt'][iseed][ielem][1]] for ielem in range(len(all_stats[key]['activ_align_init_samept'][0]))])
                        elif stat_name == 'activ_align_init_samept_alll':
                            iter_list.append([all_stats[key]['activ_align_init_samept_alll'][iseed][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_samept_alll'][0]))])
                            sub_list.append([[all_stats[key]['activ_align_init_samept_alll'][iseed][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ_align_init_samept_alll'][0]))])
                        elif stat_name == 'activ_align_init_otherpt_alll':
                            iter_list.append([all_stats[key]['activ_align_init_otherpt_alll'][iseed][ielem][0] for ielem in range(len(all_stats[key]['activ_align_init_otherpt'][0]))])
                            sub_list.append([[all_stats[key]['activ_align_init_otherpt_alll'][iseed][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ_align_init_otherpt'][0]))])
                        elif stat_name == 'activ0':
                            iter_list.append([all_stats[key][f'activ0_layer{i_l}'][iseed][ielem][0] for ielem in range(len(all_stats[key]['activ0_layer0'][0]))])
                            sub_list.append([[all_stats[key][f'activ0_layer{i_l}'][iseed][ielem][1] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['activ0_layer0'][0]))])
                        elif stat_name == 'eff_update_rank':
                            iter_list.append([all_stats[key]['delta_W_spectral_norms'][iseed][ielem][0] for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0]))])
                            try:
                                sub_list.append([[all_stats[key]['delta_W_frob_norms'][iseed][ielem][1][i_l]/all_stats[key]['delta_W_spectral_norms'][iseed][ielem][1][i_l] for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0]))])
                            except ZeroDivisionError:
                                thisappend = [[np.nan for i_l in range(nhiddenlayers+1)] for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0]))]
                                for ielem in range(len(all_stats[key]['delta_W_spectral_norms'][0])):
                                    for i_l in range(nhiddenlayers+1):
                                        try:
                                            thisappend[ielem][i_l] = all_stats[key]['delta_W_frob_norms'][iseed][ielem][1][i_l]/all_stats[key]['delta_W_spectral_norms'][iseed][ielem][1][i_l]
                                        except ZeroDivisionError:
                                            thisappend[ielem][i_l] = np.nan
                                sub_list.append(thisappend)        
                        
            plot_list.append(sub_list)

        plot_array = np.array(plot_list)
        T_final = np.max(iter_list)

        # reshape to width, layer, run
        label1 = T1 = 1
        data1 = plot_array[:,:,T1-1,:].transpose(1, 2, 0)
        label2 = T2 = 2
        data2 = plot_array[:,:,T2-1,:].transpose(1, 2, 0)
        label3 = T3 = 4
        data3 = plot_array[:,:,T3-1,:].transpose(1, 2, 0)
        label4 = T4 = T_final
        if T_final == np.max(iter_list):
            data4 = plot_array[:,:,plot_array.shape[2]-1,:].transpose(1, 2, 0)
        thisdict = {}
        colors = [adjust_lightness('tab:blue',brightness) for brightness in brightnesses[::-1]]
        
        for i_l in range(data1.shape[1]):
            if stat_name == 'delta_W_spectral_norms':
                if i_l == 0: width_expon = 1/2
                elif i_l == nhiddenlayers: width_expon = -1/2
                else: width_expon = 0
            elif stat_name == 'activation_delta':
                width_expon = 1/2 * (i_l < nhiddenlayers)
            elif stat_name in ['delta_W_x_norms', 'W_delta_x_norms']:
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
        
        this_plotpath = plot_path+f'rcc_{EXP_NAME.replace(" ","_").replace(".","")}/'
        if not os.path.exists(this_plotpath):
            os.makedirs(this_plotpath)
        if 'law' in stat_name:
            these_exponents = [0,1,1]
            for i_l in range(data1.shape[1]):
                exp1 = exponent(WIDTHS,[np.mean(data1[0,i_l,:]),np.mean(data1[-1,i_l,:])])
                print(stat_name, experiment, f' time 1, layer {i_l}, expon: ',exp1)
            for i_l in range(data1.shape[1]):
                exp2 = exponent(WIDTHS,[np.mean(data2[0,i_l,:]),np.mean(data2[-1,i_l,:])])
                print(stat_name, experiment, f' time 2, layer {i_l}, expon: ',exp2)
            for i_l in range(data1.shape[1]):
                expfin = exponent(WIDTHS,[np.mean(data4[0,i_l,:]),np.mean(data4[-1,i_l,:])])
                print(stat_name, experiment, f' time final, layer {i_l}, expon: ',expfin)
        width_plot(WIDTHS, thisdict, ylabel=y_label, titles=titles, filenam = this_plotpath+f'{stat_name}_{"nomult_" if nomultipliers else ""}'+f'{EXP_NAME}_{"_notitle" if not set_title else ""}.png',colors=colors,y_scale='log' if stat_name not in ['activ_align_init','activ_align_init_alll','eff_update_rank'] else None,exponents=these_exponents, legend_title='update step',fig_title=fig_title, ylim = [0,1] if stat_name == 'activ_align_init' else None)
        plt.clf()