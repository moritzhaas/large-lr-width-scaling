# %%
import matplotlib.pyplot as plt
from tueplots import bundles, figsizes
import numpy as np

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
onefigsize = bundles.icml2022()['figure.figsize']

brightnesses= np.linspace(0.5,1.75,4)
plot_path = 'plots/'

import string
def enumerate_subplots(axs, pos_x=-0.08, pos_y=1.05, fontsize=16):
    """Adds letters to subplots of a figure.
    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.
    Returns:
        axs (list): List of plt.axes.
    """
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())
    for n, ax in enumerate(axs.flatten()):
        ax.text(
            pos_x[n],
            pos_y[n],
            f"{string.ascii_lowercase[n]}.",
            transform=ax.transAxes,
            size=fontsize,
            weight="bold",
        )
    plt.tight_layout()
    return axs

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def adjust_fontsize(num_cols):
    # use this before creating the axes
    keys = ['font.size','axes.labelsize','legend.fontsize','xtick.labelsize','ytick.labelsize','axes.titlesize']
    for key in keys:
        plt.rcParams.update({key: bundles.icml2022()[key] * num_cols / 2})

def correct_fontsize(axs,has_legend=True, sizes=None):
    # use this for axs already created
    num_ax=len(axs)
    if sizes is None:
        sizes = num_ax
    for i in range(num_ax):
        for item in ([axs[i].title, axs[i].xaxis.label, axs[i].yaxis.label]):
            item.set_fontsize(8*sizes)
        for item in (axs[i].get_xticklabels() + axs[i].get_yticklabels()):
            item.set_fontsize(6*sizes)
        if has_legend:
            for item in axs[i].legend().get_texts():
                item.set_fontsize(6*sizes)
    return axs
    
def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


adjust_fontsize(3)



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
                            axes[iplot].text(WIDTHS[len(WIDTHS)//2], 2 * thisdata[-1,iplot,:].mean() * (np.array(WIDTHS[len(WIDTHS)//2])/WIDTHS[-1])**(exponents[iplot]),f'{np.round(exponents[iplot],3)}',color=colors[ikey], ha='center',alpha=0.99)
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
    
    plt.legend(title = legend_title,frameon=True)#title='algo')
    if fig_title is not None: fig.suptitle(fig_title)
    plt.savefig(filenam,dpi=300)
    plt.clf()

# %%