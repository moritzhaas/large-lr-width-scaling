# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
from wandb import Api
from collections import defaultdict
import matplotlib.colors as mcolors
import colorsys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
from utils.plot_utils import adjust_lightness, adjust_fontsize
#from utils.ssh_utils import get_recent_files_via_ssh, read_hdf5_keys_via_ssh, read_hdf5_entry_via_ssh, get_stats_from_h5py_via_ssh_old, get_recent_folders_via_ssh, establish_ssh_connection, execute_with_retries
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update({'text.usetex': False})
onefigsize = bundles.icml2022()['figure.figsize']
plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"mathtext.fontset": 'cm'})
def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    return (np.log(y[0])-np.log(y[-1]))/(np.log(x[0])-np.log(x[-1]))


def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of the given color by multiplying (1-luminosity) by the given amount.
    
    Args:
        color: A color string or tuple
        amount: A factor by which to adjust the lightness (0 is black, 1 is unchanged, >1 is lighter)
    
    Returns:
        Modified color with adjusted lightness
    """
    try:
        c = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, max(0, min(1, amount * l)), s)
    except ValueError:
        return color


class WandBStatsLoader:
    def __init__(self, project_name, stat_keys, exp_filter=None):
        """
        Initialize the WandB stats loader
        
        Args:
            project_name: Name of the WandB project
            stat_keys: Dictionary mapping stat names to WandB keys
            exp_filter: Optional function to filter experiments
        """
        self.api = Api()
        self.project_name = project_name
        self.stat_keys = stat_keys
        self.exp_filter = exp_filter if exp_filter else lambda config: True
        self.runs_data = []
        
    def fetch_runs(self):
        """Fetch runs from WandB and process them into a structured format"""
        runs = self.api.runs(self.project_name)
        
        for run in runs:
            config = run.config
            if not 'out_dir' in config or not self.exp_filter(config):
                continue
                
            try:
                # Extract parameters from the config
                out_dir = config['out_dir']
                width = int(out_dir.split('width=', 2)[1].split('-', 2)[0])
                lr = float(out_dir.split('lr=', 2)[1].split('-warmup', 2)[0])
                warmup = int(out_dir.split('warmup=', 2)[1].split('-', 2)[0])
                max_iters = config['eval']['max_iters']
                
                # Extract statistics - make sure to get the '_step' field to know the actual step index
                if self.stat_keys is not None:
                    stats_dict = {}
                    for stat_name, stat_key in self.stat_keys.items():
                        try:
                            # Get both the statistic and the step index
                            history = run.scan_history(keys=[stat_key, '_step'])
                            history_list = list(history)
                            
                            # Only include points where both the statistic and step are available
                            stats_dict[stat_name] = [
                                (entry['_step'], entry[stat_key]) 
                                for entry in history_list 
                                if '_step' in entry and stat_key in entry
                            ]
                        except (IndexError, KeyError):
                            continue
                else:
                    complete_history = run.scan_history()
                    history_list = list(complete_history)

                    stats_dict = {}
                    all_keys = set()
                    for entry in history_list:
                        all_keys.update(entry.keys())

                    # Remove internal keys (those starting with '_')
                    stat_keys = [key for key in all_keys if not key.startswith('_') and key != 'global_step']

                    for stat_key in stat_keys:
                        stats_dict[stat_key] = [
                            (entry['_step'], entry[stat_key])
                            for entry in history_list
                            if '_step' in entry and stat_key in entry and not pd.isna(entry[stat_key])
                        ]

                    print(f"Found {len(stats_dict)} statistics")

                # Add run info and stats to the dataset
                run_info = {
                    'run_name': run.name,
                    'width': width,
                    'lr': lr,
                    'warmup': warmup,
                    'max_iters': max_iters,
                    'stats': stats_dict
                }
                self.runs_data.append(run_info)
                
                print(f"\nWandB Name: {run.name}")
                print(f"Width: {width}, LR: {lr}, Warmup: {warmup}")
                
            except (IndexError, KeyError, ValueError) as e:
                print(f"Error processing run {run.name}: {e}")
                continue
                
        return self.runs_data
    
    def to_dataframe(self):
        """Convert the runs data to a pandas DataFrame format suitable for plotting"""
        if not self.runs_data:
            self.fetch_runs()
        
        data_records = []
        
        for run in self.runs_data:
            width = run['width']
            lr = run['lr']
            warmup = run['warmup']
            
            for stat_name, stat_values in run['stats'].items():
                for step, value in stat_values:
                    data_records.append({
                        'width': width,
                        'lr': lr,
                        'warmup': warmup,
                        'step': step,  # This is now the actual step from WandB
                        'statistic': stat_name,
                        'value': value,
                        'run_name': run['run_name']
                    })
        
        return pd.DataFrame(data_records)
    
    def get_best_lr_for_widths(self, metric='loss', step=None, widths=None, strategy='min'):
        """
        Find the best learning rate for each width based on a metric
        
        Args:
            metric: The metric to optimize (e.g., 'loss')
            step: The step at which to evaluate the metric (if None, uses the last available step)
            widths: List of widths to analyze (if None, uses all available)
            strategy: 'min' to minimize the metric, 'max' to maximize it
        
        Returns:
            Dictionary mapping width to best learning rate
        """
        df = self.to_dataframe()
        
        # Filter by metric
        df_filtered = df[df['statistic'] == metric]
        
        # Filter by widths if specified
        if widths is not None:
            df_filtered = df_filtered[df_filtered['width'].isin(widths)]
        
        width_to_lr = {}
        
        # For each width
        for width in df_filtered['width'].unique():
            width_data = df_filtered[df_filtered['width'] == width]
            
            # Determine the step to evaluate
            if step is None:
                # Use the last available step for each run
                runs_with_last_step = width_data.groupby('run_name')['step'].max().reset_index()
                runs_with_last_step = runs_with_last_step.rename(columns={'step': 'last_step'})
                
                # Merge to get the values at the last step
                merged_data = pd.merge(
                    width_data, 
                    runs_with_last_step, 
                    on='run_name'
                )
                eval_data = merged_data[merged_data['step'] == merged_data['last_step']]
            else:
                # Find the closest step for each run
                eval_data = pd.DataFrame()
                for run_name, run_data in width_data.groupby('run_name'):
                    if not run_data.empty:
                        idx = (run_data['step'] - step).abs().idxmin()
                        eval_data = pd.concat([eval_data, run_data.loc[[idx]]])
            
            # Group by learning rate and calculate average metric
            lr_performance = eval_data.groupby('lr')['value'].mean().reset_index()
            
            # Find the best learning rate
            if strategy == 'min':
                best_lr = lr_performance.loc[lr_performance['value'].idxmin()]['lr']
            else:
                best_lr = lr_performance.loc[lr_performance['value'].idxmax()]['lr']
            
            width_to_lr[width] = best_lr
        
        return width_to_lr


class WandBStatsPlotter:
    def __init__(self, dataframe):
        """
        Initialize the plotter with a DataFrame of stats
        
        Args:
            dataframe: Pandas DataFrame with the stats to plot
        """
        self.df = dataframe
        
    def plot_across_widths(self, statistic, steps=[None], lr_by_width = None, warmup=None, find_closest_step=False,
                          log_x=True, log_y=True,title='SP',y_label=None,lr_exponent=None, text_exponents=None,ylim = None,notitle=False,nolegend=False):
        """
        Plot a statistic across different widths for a specific step
        
        Args:
            statistic: Name of the statistic to plot
            step: The training step to plot (if None, uses the last available step)
            lr: Filter by learning rate (optional)
            warmup: Filter by warmup steps (optional)
            find_closest_step: If True and exact step not found, use the closest available step
            log_x: Whether to use log scale for x-axis
            log_y: Whether to use log scale for y-axis
        """
        plt.figure(figsize=(1*onefigsize[0],1*onefigsize[1]))
        n_steps = len(steps)
        colors =sns.color_palette("rocket_r", n_colors=n_steps)
        #colors = plt.cm.viridis(np.linspace(0, 0.8, n_steps))       
        legend_elements = []
        
        # Filter the dataframe
        df_filtered = self.df[self.df['statistic'] == statistic]
        if lr_by_width is not None:
            mask = pd.Series(False, index=df_filtered.index)
            for width, lr in lr_by_width.items():
                mask = mask | ((df_filtered['width'] == width) & (df_filtered['lr'] == lr))

            df_filtered = df_filtered[mask]
            #df_filtered = df_filtered[df_filtered['lr'] == lr]
        
        if warmup is not None:
            df_filtered = df_filtered[df_filtered['warmup'] == warmup]
        
        for istep, step in enumerate(steps):
            step_color = colors[istep]
            if step is None:
                # Get the last step for each run
                last_steps = df_filtered.groupby(['width', 'run_name'])['step'].max().reset_index()
                #print(f"Last steps: {last_steps}")
                step = last_steps['step'].max()
                df_merged = pd.merge(df_filtered, last_steps, on=['width', 'run_name', 'step'])
            else:
                if find_closest_step:
                    # Find the closest step for each run
                    df_closest = pd.DataFrame()
                    for (w, run), group in df_filtered.groupby(['width', 'run_name']):
                        if not group.empty:
                            # Find index of minimum distance to target step
                            closest_idx = (group['step'] - step).abs().idxmin()
                            closest_row = group.loc[[closest_idx]]
                            df_closest = pd.concat([df_closest, closest_row])
                    df_merged = df_closest
                else:
                    # Use exact step only
                    df_merged = df_filtered[df_filtered['step'] == step]
            
            # Group by width and calculate statistics
            width_stats = df_merged.groupby('width')['value'].agg(['mean', 'std']).reset_index()
            
            # Plot mean with error bars
            plt.errorbar(width_stats['width'], width_stats['mean'], yerr=width_stats['std'], color=step_color,
                        marker='o', linestyle='-', capsize=5)
            
            # Add individual points colored by run
            sns.scatterplot(data=df_merged, x='width', y='value', hue='step',color=step_color, alpha=0.6)
            if text_exponents is not None:
                thisexpon = exponent(width_stats['width'].to_list(), width_stats['mean'].to_list())
                plt.text(width_stats['width'].to_list()[1],width_stats['mean'].to_list()[1]*text_exponents[istep],f'{np.round(thisexpon,3)}',color=step_color, ha='center')

            if not nolegend:
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], color=step_color, marker='o', linestyle='-',
                        label=f'{step}', markerfacecolor=step_color)
                )
        if ylim is not None:
            plt.ylim(ylim[0],ylim[1])

        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
            
        plt.xlabel('Width')
        plt.ylabel(statistic if y_label is None else y_label)
        #title = f'{statistic} across Widths'
        title = title if title is not None else f'{statistic}'
        # if step is not None:
        #     if find_closest_step:
        #         title += f' near step {step}'
        #     else:
        #         title += f' at step {step}'
        if title == f'{statistic}':
            if lr_by_width is not None:
                for width, lr in lr_by_width.items(): break
                title += f', LR={lr}'
            if lr_exponent is not None:
                title += f'*(width/256)**{lr_exponent}'
        # if warmup is not None:
        #     title += f', Warmup={warmup}'
        if not notitle:
            plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(list(lr_by_width.keys()),labels=list(lr_by_width.keys()))
        if not nolegend: plt.legend(handles=legend_elements, title='Training Step')#plt.legend(title='Step')#bbox_to_anchor=(1.05, 1), loc='upper left')
        if nolegend: plt.gca().get_legend().remove()
        plt.tight_layout()
        return plt
    
    def plot_over_time(self, statistic, widths=None, lr_by_width=None, warmup=None, align_steps=False,height=1,
                      log_x=True, log_y=True, base_color='tab:blue',title=None,title_font=None,y_label=None,ylim=None,legend_title=None,legend_labels = None, expons=None, text_pos=None):
        """
        Plot a statistic over time with different lines for different widths
        
        Args:
            statistic: Name of the statistic to plot (string or list of statistics)
            widths: List of widths to include (if None, uses all)
            lr_by_width: Dictionary mapping width to learning rate to use
            warmup: Filter by warmup steps (optional)
            align_steps: If True, interpolate to ensure all runs have values at the same steps
            log_x: Whether to use log scale for x-axis
            log_y: Whether to use log scale for y-axis
            base_color: Base color to use, will be darkened for larger widths
        """
        plt.figure(figsize=(1*onefigsize[0],height*onefigsize[1]))
        
        # Convert statistic to a list if it's not already
        if isinstance(statistic, str):
            statistics = [statistic]
        else:
            statistics = statistic
            
        # For each statistic
        for stat_idx, stat in enumerate(statistics):
            # Filter the dataframe for this statistic
            df_filtered = self.df[self.df['statistic'] == stat]
            if warmup is not None:
                df_filtered = df_filtered[df_filtered['warmup'] == warmup]
            
            # Filter by widths if specified
            if widths is not None:
                df_filtered = df_filtered[df_filtered['width'].isin(widths)]
                unique_widths = sorted(width for width in widths if width in df_filtered['width'].unique())
            else:
                unique_widths = sorted(df_filtered['width'].unique())
            
            # Calculate brightness values that get darker for larger widths
            brightnesses = np.linspace(0.5,1.75, len(unique_widths))[::-1]
            
            # For each width
            for i, width in enumerate(unique_widths):
                # Apply the width-specific learning rate if provided
                width_data = df_filtered[df_filtered['width'] == width]
                if lr_by_width is not None and width in lr_by_width:
                    width_data = width_data[width_data['lr'] == lr_by_width[width]]
                
                # Skip if no data
                if width_data.empty:
                    continue
                
                # Get color for this width and statistic
                if len(statistics) > 1:
                    # If multiple statistics, use different base colors for each
                    stat_colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange']
                    base = stat_colors[stat_idx % len(stat_colors)]
                else:
                    base = base_color
                    
                # Adjust darkness based on width (larger widths get darker colors)
                color = adjust_lightness(base, brightnesses[i])
                
                if align_steps:
                    # This would require interpolation to align steps across runs
                    # First, get all unique runs for this width
                    runs = width_data['run_name'].unique()
                    
                    # For each run, get the data and interpolate
                    all_runs_data = []
                    for run_name in runs:
                        run_data = width_data[width_data['run_name'] == run_name].sort_values('step')
                        
                        # Skip if not enough data points
                        if len(run_data) < 2:
                            continue
                            
                        # Create interpolation function
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(
                            run_data['step'], 
                            run_data['value'],
                            bounds_error=False,
                            fill_value=(run_data['value'].iloc[0], run_data['value'].iloc[-1])
                        )
                        
                        # Store for later use
                        all_runs_data.append((run_name, interp_func, run_data['step'].min(), run_data['step'].max()))
                    
                    if all_runs_data:
                        # Create a common step grid
                        min_step = max(data[2] for data in all_runs_data)
                        max_step = min(data[3] for data in all_runs_data)
                        
                        if min_step <= max_step:
                            common_steps = np.linspace(min_step, max_step, 100)
                            
                            # Interpolate each run to this grid
                            interpolated_values = np.zeros((len(all_runs_data), len(common_steps)))
                            for j, (_, interp_func, _, _) in enumerate(all_runs_data):
                                interpolated_values[j] = interp_func(common_steps)
                            
                            # Calculate mean and std
                            mean_values = np.mean(interpolated_values, axis=0)
                            std_values = np.std(interpolated_values, axis=0)
                            
                            # Plot
                            #label = f'{stat.split("_",2)[-1]}, {width}' if len(statistics) > 1 else f'{width}'
                            if legend_labels is None:
                                label = f'{stat.split("_",2)[-1]}, {width}' if len(statistics) > 1 else f'{width}'
                            else:
                                label = legend_labels[stat_idx] if i == 1 else None
                            plt.plot(common_steps, mean_values, label=label, color=color)
                            plt.fill_between(
                                common_steps, 
                                mean_values - std_values,
                                mean_values + std_values,
                                alpha=0.2, 
                                color=color
                            )
                else:
                    # Group by step and calculate mean and std directly
                    # This might have gaps where some runs don't have data at certain steps
                    step_stats = width_data.groupby('step')['value'].agg(['mean', 'std']).reset_index()
                    
                    # Plot mean line with shaded error region
                    if legend_labels is None:
                        label = f'{stat.split("_",2)[-1]}, {width}' if len(statistics) > 1 else f'{width}'
                    else:
                        if legend_title =='width':
                            label = width if stat_idx==0 else None
                        else:
                            label = legend_labels[stat_idx] if i == 1 else None
                    plt.plot(step_stats['step'], step_stats['mean'], label=label, color=color)
                    
                    if expons is not None and i==len(unique_widths)-1:
                        plt.text(text_pos[stat_idx][0],text_pos[stat_idx][1],expons[stat_idx],color=color, ha='center',fontsize=14,fontweight='bold')


                    # Only add error bands if we have multiple runs
                    if width_data['run_name'].nunique() > 1:
                        plt.fill_between(
                            step_stats['step'], 
                            step_stats['mean'] - step_stats['std'],
                            step_stats['mean'] + step_stats['std'],
                            alpha=0.2, 
                            color=color
                        )
        
        if ylim is not None:
            plt.ylim(ylim[0],ylim[1])
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
            
        plt.xlabel('Training Step')
        if y_label is not None:
            plt.ylabel(y_label)
        
        #title = ', '.join(statistics) if len(statistics) > 1 else statistics[0]
        #title += ' over Time'
        # if lr_by_width is not None:
        #     title += f', Width-specific LRs'
        if title_font is None:
            plt.title(title)
        else:
            plt.title(title,fontsize = title_font)
        plt.grid(True, alpha=0.3)
        if legend_title is None:
            plt.legend(title='Layer, Width' if len(statistics) > 1 else 'Width',frameon=True,loc='lower left')
        else:
            plt.legend(title=legend_title,frameon=True,loc='lower left')
        plt.tight_layout()
        
        return plt
        
    # def plot_all_deltaWx_over_time(self, widths=None, lr_by_width=None, warmup=None, 
    #                               log_x=True, log_y=True):
    #     """
    #     Plot all DeltaWx statistics over time
        
    #     Args:
    #         widths: List of widths to include
    #         lr_by_width: Dictionary mapping width to learning rate to use
    #         warmup: Filter by warmup steps (optional)
    #         log_x: Whether to use log scale for x-axis
    #         log_y: Whether to use log scale for y-axis
    #     """
    #     # Get all DeltaWx statistics
    #     deltaWx_stats = [stat for stat in self.df['statistic'].unique() if stat.startswith('DeltaWx')]
        
    #     return self.plot_over_time(
    #         statistic=deltaWx_stats,
    #         widths=widths,
    #         lr_by_width=lr_by_width,
    #         warmup=warmup,
    #         align_steps=True,
    #         log_x=log_x,
    #         log_y=log_y
    #     )
    
    def get_available_steps(self, statistic, width=None, lr=None, warmup=None):
        """
        Get a list of available steps for the given parameters
        
        Args:
            statistic: Name of the statistic to check
            width: Filter by width (optional)
            lr: Filter by learning rate (optional)
            warmup: Filter by warmup steps (optional)
        
        Returns:
            List of available step values
        """
        df_filtered = self.df[self.df['statistic'] == statistic]
        if width is not None:
            df_filtered = df_filtered[df_filtered['width'] == width]
        if lr is not None:
            df_filtered = df_filtered[df_filtered['lr'] == lr]
        if warmup is not None:
            df_filtered = df_filtered[df_filtered['warmup'] == warmup]
            
        return sorted(df_filtered['step'].unique())

def add_stat_to_df(df, stat_name, stat_fct, required_stats):
    """
    Add a new statistic to the DataFrame.
    
    Args:
        df: The original DataFrame
        stat_name: Name of the new statistic
        stat_value: Value of the new statistic
    """

    df_pivot = df.pivot_table(
        index=['width', 'lr', 'warmup', 'step', 'run_name'],
        columns='statistic',
        values='value'
    ).reset_index()

    # Calculate the new statistic but only for rows that have all required values

    # Check if all required statistics are present
    if all(stat in df_pivot.columns for stat in required_stats):
        # Filter out rows with NaN values in any of the required columns
        mask = df_pivot[required_stats].notna().all(axis=1)
        df_valid = df_pivot[mask].copy()
        
        # Calculate the new statistic only for valid rows
        df_valid[stat_name] = stat_fct(df_valid) #np.log(
        #     df_valid['width'] * df_valid['DeltaWx_out'] / 
        #     (df_valid['DeltaW_frob_out'] * df_valid['activation_norm_lnout'])
        # ) / np.log(df_valid['width'])
        
        # Convert back to original format
        new_stat = df_valid[['width', 'lr', 'warmup', 'step', 'run_name', stat_name]].copy()
        new_stat['statistic'] = stat_name
        new_stat = new_stat.rename(columns={stat_name: 'value'})
        
        # Append the new statistic to the original dataframe
        df_with_new_stat = pd.concat([df, new_stat[['width', 'lr', 'warmup', 'step', 'statistic', 'value', 'run_name']]])
        
    else:
        print("Error: Not all required statistics found in the DataFrame")
        df_with_new_stat = df.copy()  # Return original if missing required stats
    return df_with_new_stat


# %%
# # Example usage
# if __name__ == "__main__":
plot_path = './figures/litgpt_rcc/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

layername_from_statkey = {'lm_head':'out','ln_f':'lnout','wte':'embed','mlp.fc':'mlp1','mlp.proj':'mlp2'}

# Define the statistics to extract
stats_dict = {
    'W_norm_out':'parameter/lm_head.weight/l2norm',
    'W_opnorm_out':'parameter/lm_head.weight/opnorm',
    'W_norm_lnout':'parameter/transformer.ln_f.weight/l2norm',
    'W_opnorm_lnout':'parameter/transformer.ln_f.weight/opnorm',
    'W_norm_mlp':'parameter/transformer.h.2.mlp.fc.weight/l2norm',
    'W_opnorm_mlp':'parameter/transformer.h.2.mlp.fc.weight/opnorm',
    'DeltaWx_out': '(W_t-W_0)x_t/lm_head.weight/l2norm',
    'DeltaWx_lnout': '(W_t-W_0)x_t/transformer.ln_f.weight/l2norm',
    'DeltaWx_wte': '(W_t-W_0)x_t/transformer.wte.weight/l2norm',
    'DeltaWx_mlp':'(W_t-W_0)x_t/transformer.h.2.mlp.fc.weight/l2norm',
    'DeltaWx_norm1':'(W_t-W_0)x_t/transformer.h.2.norm_1.weight/l2norm',
    'DeltaWx_knorm':'(W_t-W_0)x_t/transformer.h.2.attn.k_norm.weight/l2norm',
    'DeltaWx_proj':'(W_t-W_0)x_t/transformer.h.2.attn.proj.weight/l2norm',
    'activation_norm_out': 'activation/lm_head/l2norm',
    'activation_norm_lnout': 'activation/transformer.ln_f/l2norm',
    'activation_norm_wte': 'activation/transformer.wte/l2norm',
    'activation_norm_norm2': 'activation/transformer.h.2.norm_2/l2norm',
    'activation_norm_mlp': 'activation/transformer.h.2.mlp.fc/l2norm',
    'activation_norm_norm1': 'activation/transformer.h.2.norm_1/l2norm',
    'activation_norm_knorm': 'activation/transformer.h.2.attn.k_norm/l2norm',
    'activation_norm_proj': 'activation/transformer.h.2.attn.proj/l2norm',
    'activation_diff_out': 'activation_difference/lm_head/l2norm',
    'activation_diff_lnout': 'activation_difference/transformer.ln_f/l2norm',
    'activation_diff_wte': 'activation_difference/transformer.wte/l2norm',
    'activation_diff_norm2':'activation_difference/transformer.h.2.norm_2/l2norm',
    'activation_diff_mlp':'activation_difference/transformer.h.2.mlp.fc/l2norm',
    'activation_diff_norm1':'activation_difference/transformer.h.2.norm_1/l2norm',
    'activation_diff_knorm':'activation_difference/transformer.h.2.attn.k_norm/l2norm',
    'activation_diff_proj':'activation_difference/transformer.h.2.attn.proj/l2norm',
    'activation_norm_in_out': f'(W_t-W_0)x_t/lm_head/x_t/l2norm',
    'activation_norm_in_lnout': f'(W_t-W_0)x_t/transformer.ln_f/x_t/l2norm',
    'activation_norm_in_wte': f'(W_t-W_0)x_t/transformer.wte/x_t/l2norm',
    'activation_norm_in_mlp': f'(W_t-W_0)x_t/transformer.h.2.mlp.fc/x_t/l2norm',
    'activation_norm_in_norm1': f'(W_t-W_0)x_t/transformer.h.2.norm_1/x_t/l2norm',
    'activation_norm_in_knorm': f'(W_t-W_0)x_t/transformer.h.2.attn.k_norm/x_t/l2norm',
    'activation_norm_in_proj': f'(W_t-W_0)x_t/transformer.h.2.attn.proj/x_t/l2norm',
    'DeltaW_frob_out': 'parameter_difference/lm_head.weight/l2norm',
    'DeltaW_op_out': 'parameter_difference/lm_head.weight/opnorm',
    'DeltaW_frob_lnout': 'parameter_difference/transformer.ln_f.weight/l2norm',
    'DeltaW_op_lnout': 'parameter_difference/transformer.ln_f.weight/opnorm',
    'DeltaW_frob_mlp': 'parameter_difference/transformer.h.2.mlp.fc.weight/l2norm',
    'DeltaW_op_mlp': 'parameter_difference/transformer.h.2.mlp.fc.weight/opnorm',
    'DeltaW_frob_norm1': 'parameter_difference/transformer.h.2.norm_1/l2norm',
    'DeltaW_op_norm1': 'parameter_difference/transformer.h.2.norm_1/opnorm',
    'DeltaW_frob_knorm': 'parameter_difference/transformer.h.2.attn.k_norm/l2norm',
    'DeltaW_op_knorm': 'parameter_difference/transformer.h.2.attn.k_norm/opnorm',
    'DeltaW_frob_proj': 'parameter_difference/transformer.h.2.attn.proj/l2norm',
    'DeltaW_op_proj': 'parameter_difference/transformer.h.2.attn.proj/opnorm',
    'WDeltax_wte':'W_0(x_t-x_0)/transformer.wte/l2norm',
    'WDeltax_lnout':'W_0(x_t-x_0)/transformer.ln_f.weight/l2norm',
    'WDeltax_out':'W_0(x_t-x_0)/lm_head.weight/l2norm',
    'WDeltax_mlp':'W_0(x_t-x_0)/transformer.h.2.mlp.fc.weight/l2norm',
    'WDeltax_norm1':'W_0(x_t-x_0)/transformer.h.2.norm_1.weight/l2norm',
    'WDeltax_knorm':'W_0(x_t-x_0)/transformer.h.2.attn.k_norm.weight/l2norm',
    'WDeltax_proj':'W_0(x_t-x_0)/transformer.h.2.proj.weight/l2norm',
    'Deltax_out': 'W_0(x_t-x_0)/lm_head/x_t-x_0/l2norm',
    'Deltax_lnout':'W_0(x_t-x_0)/transformer.ln_f/x_t-x_0/l2norm',
    'Deltax_mlp':'W_0(x_t-x_0)/transformer.h.2.mlp.fc/x_t-x_0/l2norm',
    'Deltax_norm1':'W_0(x_t-x_0)/transformer.h.2.norm_1/x_t-x_0/l2norm',
    'Deltax_knorm':'W_0(x_t-x_0)/transformer.h.2.attn.k_norm/x_t-x_0/l2norm',
    'Deltax_proj':'W_0(x_t-x_0)/transformer.h.2.attn.proj/x_t-x_0/l2norm',
    'Deltax_norm_lnout':'activation_difference/transformer.ln_f/l2norm',
    'Deltax_norm_norm2':'activation_difference/transformer.h.2.norm_2/l2norm',
    'loss': 'loss'
}
widths_to_analyze = [256, 1024, 4096]

# Function to filter experiments (customize as needed)
def filter_experiments(config):
    return 'warmup=700-id=2025' in config['out_dir']

project_name="mup_limitations/standard-transformer-coordinate-check-init"
#project_name="mup_limitations/mup-adamw-coordinate-check-init"
#project_name="mup_limitations/mup-sp-init-coordinate-check"

if 'mup-sp-init-coordinate-check' in project_name:
    exp_short='spfullalign'
elif 'mup-adamw' in project_name:
    exp_short = 'mupadam'
elif 'standard-transformer-coordinate' in project_name:
    exp_short = 'spadam'


# Load data
loader = WandBStatsLoader(
    project_name=project_name, #standard-transformer-coordinate-check-init  #mup-adamw-coordinate-check-init
    stat_keys=stats_dict,
    exp_filter=filter_experiments
)

# Fetch runs and convert to DataFrame
runs_data = loader.fetch_runs()
df = loader.to_dataframe()

# derive other stats

def l2_to_rms(stat, width):
    return stat / np.sqrt(width)

d_in, base_hidden_size, mlp_hiddenmult, d_out = 50304, 256, 4, 50304

if 'standard' in project_name:
    base_lr = 0.01
    lrexpon = -1
else:
    base_lr = 0.0031622776601683794
    lrexpon = 0
lr_by_width = {width: base_lr * (width/256)**lrexpon for width in widths_to_analyze}

# df['Delta_Wx_rms_out'] = df['DeltaWx_out'] / np.sqrt(d_out)
# df['Delta_Wx_rms_out'] = df['DeltaWx_out'] / np.sqrt(d_out)

# %%
# for now just for ln_f RMS norms for refined coord check mainplot: activation_norm_lnout, activation_diff_lnout, DeltaWx_lnout, WDeltax_lnout
layer = 'wte'
layers = ['mlp','lnout','out','knorm','norm1', 'proj','wte'] #['wte','proj','knorm','norm1'] # 'mlp','lnout','out'
for layer in layers:
    for statname in ['activation_norm', 'activation_diff', 'DeltaWx', 'WDeltax']:
        this_name = f'{statname}_rms_{layer}'
        required_stats = ['width', f'{statname}_{layer}']
        
        def this_fct(x):
            if layer == 'out':
                fanout=d_out
            elif layer == 'mlp':
                fanout = mlp_hiddenmult*x['width']
            else:
                fanout = x['width']
            return x[f'{statname}_{layer}'] / np.sqrt(fanout)

        df = add_stat_to_df(df, this_name, this_fct, required_stats)

    
    print("Best learning rate for each width:")
    for width, lr in lr_by_width.items():
        print(f"Width {width}: LR = {lr}")

    # Create plotter
    adjust_fontsize(3)
    plotter = WandBStatsPlotter(df)
    othersteps= True
    target_steps = [2,10,100,700] if othersteps else [1,4,10,100]
    for statname in ['activation_norm', 'activation_diff', 'DeltaWx', 'WDeltax']:
        this_name = f'{statname}_rms_{layer}'
        text_exponents = [1.35,0.3,0.4,1.35] if othersteps else [1.35,0.5,0.5,1.35]
        ylim = [9.99e-3,3] if layer=='lnout' else [9.99e-3,30]
        if 'DeltaWx' in this_name:
            text_exponents = [0.3,0.3,1.7,1.7]
            y_label = r'$||\Delta W_t x_t||_{RMS}$'
            title = 'Effective updates'
            ylim=None
        elif 'activation_norm' in this_name:
            text_exponents = [0.5,0.3,0.18,0.1] #[1.02,1.03,1.04,1.04]
            y_label = r'$||W_t x_t||_{RMS}$'
            title = 'Activation norm'
            #ylim = [9.99e-3,3]
        elif 'activation_diff' in this_name:
            y_label = r'$||\Delta (W_t x_t)||_{RMS}$'
            title = 'Activation updates'
            #ylim = [9.99e-3,3]
        elif 'WDeltax' in this_name:
            y_label = r'$||W_0 \Delta x_t||_{RMS}$'
            title = 'Propagating updates'
            #ylim = [9.99e-3,3]
        #else: #if 'activation' in this_name:
        if layer == 'wte' or 'mup-adamw-coordinate-check-init' in project_name: ylim=None
        title=None
        plt_width = plotter.plot_across_widths(statistic=this_name, steps=target_steps,lr_by_width=lr_by_width, lr_exponent=lrexpon,log_y=True,text_exponents=text_exponents,y_label=y_label,title=title,ylim=ylim,notitle=True,nolegend=True)# for expon 0: text_exponents=[0.96,1.02,0.96,1.02])
        plt_width.savefig(plot_path+f'{this_name}_across_widths_{exp_short}_lr{np.round(base_lr,5)}_{lrexpon}_mainfigs{"_otherothersteps" if othersteps else ""}_largefont.png')



# %%
# quotient W0 Delta x
adjust_fontsize(2.75)    

def create_opnorm_quotient(df, layer):
    # Pre-compute the W0_op_norms dictionary for all widths that appear in the data
    unique_widths = df['width'].unique()
    W0_op_norms = {}
    
    for width in unique_widths:
        op_norm_rows = df[(df['width'] == width) & 
                          (df['statistic'] == f'W_opnorm_{layer}') & 
                          (df['step'] == 0)]
        if not op_norm_rows.empty:
            W0_op_norms[width] = op_norm_rows['value'].iloc[0]
    
    # Create a pivot table to get required values for each (width, lr, warmup, step) combination
    pivot_df = df.pivot_table(
        index=['width', 'lr', 'warmup', 'step'],
        columns='statistic',
        values='value'
    )
    
    # Define a function to calculate the quotient for each row
    def quotient_func(row):
        width = row.name[0]  # width is the first element of the MultiIndex
        if width in W0_op_norms and f'WDeltax_{layer}' in row and f'Deltax_{layer}' in row:
            return row[f'WDeltax_{layer}'] / (W0_op_norms[width] * row[f'Deltax_{layer}'])
        return None
    
    # Apply the function to each row
    pivot_df[f'WDeltax_op_quotient_{layer}'] = pivot_df.apply(quotient_func, axis=1)
    
    # Reset the index to convert back to regular DataFrame
    pivot_df = pivot_df.reset_index()
    
    # Create a new DataFrame with the derived statistic
    derived_rows = pivot_df[['width', 'lr', 'warmup', 'step', f'WDeltax_op_quotient_{layer}']].dropna(
        subset=[f'WDeltax_op_quotient_{layer}']
    ).copy()
    derived_rows['statistic'] = f'WDeltax_op_quotient_{layer}'
    derived_rows = derived_rows.rename(columns={f'WDeltax_op_quotient_{layer}': 'value'})
    
    # Append the new rows to the original DataFrame
    result_df = pd.concat([df, derived_rows], ignore_index=True)
    
    return result_df

df_result = create_opnorm_quotient(df,'out')
df_result = create_opnorm_quotient(df_result,'lnout')
df_result = create_opnorm_quotient(df_result,'mlp')

def create_rmsnorm_quotient(df, layer):
    # Pre-compute the W0_op_norms dictionary for all widths that appear in the data
    unique_widths = df['width'].unique()
    W0_norms = {}
    
    for width in unique_widths:
        op_norm_rows = df[(df['width'] == width) & 
                          (df['statistic'] == f'W_norm_{layer}') & 
                          (df['step'] == 0)]
        if not op_norm_rows.empty:
            W0_norms[width] = op_norm_rows['value'].iloc[0]

    # W_norm_out: width indep
    # width=1024-lr=0.0025-warmup=700-id=20250507222726	141.8572540283203
    # width=1024-lr=0.01-warmup=700-id=20250507222524	141.8572540283203
    # width=4096-lr=0.000625-warmup=700-id=20250507122925	141.85214233398438
    # width=4096-lr=0.01-warmup=700-id=20250507122826	141.85214233398438
    # width=256-lr=0.01-warmup=700-id=20250507222441	141.81288146972656
    
    # Create a pivot table to get required values for each (width, lr, warmup, step) combination
    pivot_df = df.pivot_table(
        index=['width', 'lr', 'warmup', 'step'],
        columns='statistic',
        values='value'
    )

    
    # Define a function to calculate the quotient for each row
    def quotient_func(row):
        width = row.name[0]  # width is the first element of the MultiIndex
        
        factor = 1 if layer == 'out' else width  **-1/2 #RMS needs n, but then we divide by the expected exponent

        if width in W0_norms and f'WDeltax_{layer}' in row and f'Deltax_{layer}' in row:
            return factor * row[f'WDeltax_{layer}'] / (W0_norms[width] * row[f'Deltax_{layer}'])
        return None
    
    # Apply the function to each row
    pivot_df[f'WDeltax_rms_quotient_{layer}'] = pivot_df.apply(quotient_func, axis=1)
    
    # Reset the index to convert back to regular DataFrame
    pivot_df = pivot_df.reset_index()
    
    # Create a new DataFrame with the derived statistic
    derived_rows = pivot_df[['width', 'lr', 'warmup', 'step', f'WDeltax_rms_quotient_{layer}']].dropna(
        subset=[f'WDeltax_rms_quotient_{layer}']
    ).copy()
    derived_rows['statistic'] = f'WDeltax_rms_quotient_{layer}'
    derived_rows = derived_rows.rename(columns={f'WDeltax_rms_quotient_{layer}': 'value'})
    
    # Append the new rows to the original DataFrame
    result_df = pd.concat([df, derived_rows], ignore_index=True)
    
    return result_df

df_result2= create_rmsnorm_quotient(df,'out')
df_result2 = create_rmsnorm_quotient(df_result2,'lnout')
df_result2 = create_rmsnorm_quotient(df_result2,'mlp')


plotter = WandBStatsPlotter(df_result)
plotter_rms = WandBStatsPlotter(df_result2)
ylog = True

for statname, ylim, title, layer, _ in zip(
    ['WDeltax_rms_quotient_out','WDeltax_op_quotient_out','WDeltax_rms_quotient_lnout', 'WDeltax_op_quotient_lnout'],
    [None,None,None,None],#[[0.49,1.01],[0.49,1.01],[-0.51,0.01],[-0.51,0.01]],
    [r'$||W_0 \Delta x_t||/(||W_0||_{RMS}||\Delta x_t||)$',r'$||W_0 \Delta x_t||/(||W_0||_{op}||\Delta x_t||)$',r'$||W_0 \Delta x_t||/(||W_0||_{RMS}||\Delta x_t||)$',r'$||W_0 \Delta x_t||/(||W_0||_{op}||\Delta x_t||)$'],
    ['out','out','lnout','lnout'],
    ['RMS-alignment($W_0, \Delta x_t)$, outlayer', 'Operator-alignment($W_0, \Delta x_t)$, outlayer','RMS-alignment($W_0, \Delta x_t)$, last LN', 'Operator-alignment($W_0, \Delta x_t)$, last LN']):
    #if 'rms' in statname: continue
    expons,text_pos = None,None
    if layer == 'out':
        if ylog:
            ylim= [0.09,1.01]
        else:
            ylim = [-0.01,1.01]
        if '_rms_' not in statname:
            expons = ['-0.064']
            text_pos = [(1000,0.5)]
    else:
        ylim = [9.99e-4,2.01]
    if '_rms_' in statname:
        ylim=None
        plt_deltaWx = plotter_rms.plot_over_time(
            statistic=statname, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,height=1.2) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statname}_{base_lr}_{lrexpon}_{exp_short}_mainfig_test_largefont_high.png',dpi=300)
        plt.clf()
    else:
        plt_deltaWx = plotter.plot_over_time(
            statistic=statname, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,expons=expons,text_pos = text_pos,height=1.2) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statname}_{base_lr}_{lrexpon}_{exp_short}_mainfig_test_largefont_high.png',dpi=300)
        plt.clf()

if 'standard' in project_name:
    expons = ('-0.16', '-0.04', '-0.10')
    text_pos = ((1750,0.17),(1750,0.28),(1750,0.45))
else:
    expons, text_pos = None,None

for statnames, ylim, title, _ in zip(
    [['WDeltax_rms_quotient_lnout','WDeltax_rms_quotient_mlp','WDeltax_rms_quotient_out'],['WDeltax_op_quotient_lnout','WDeltax_op_quotient_mlp','WDeltax_op_quotient_out']],
    [None,None,None,None],#[[0.49,1.01],[0.49,1.01],[-0.51,0.01],[-0.51,0.01]],
    [r'$||W_0 \Delta x_t||/(||W_0||_{RMS}||\Delta x_t||)$',r'$||W_0 \Delta x_t||/(||W_0||_{op}||\Delta x_t||)$'],
    [r'RMS-alignment($W_0,\Delta x_t)$', r'Operator-alignment($W_0,\Delta x_t)$']):
    # if 'rms' in statnames[0]: continue
    # if layer == 'out':
    #     if ylog:
    #         ylim= [0.09,1.01]
    #     else:
    #         ylim = [-0.01,1.01]
    # else:
    #     ylim = [9.99e-4,2.01]
    legend_title = 'width' if not '_op_' in statnames[0] else 'layer'

    if '_rms_' in statnames[0]:
        plt_deltaWx = plotter_rms.plot_over_time(
            statistic=statnames, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim,height=1.2, y_label = None,title=title,legend_title=legend_title,legend_labels=['last LN','mlp hidden','outlayer'], title_font=14) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statnames[0]}_{base_lr}_{lrexpon}_{exp_short}_mainfig_multilayer_test_largefont_largetitle_high.png',dpi=300)
        plt.clf()
    else:
        plt_deltaWx = plotter.plot_over_time(
            statistic=statnames, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim,height=1.2, y_label = None,title=title,legend_title=legend_title,legend_labels=['last LN','mlp hidden','outlayer'],expons=expons,text_pos = text_pos, title_font=14) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statnames[0]}_{base_lr}_{lrexpon}_{exp_short}_mainfig_multilayer_test_largefont_largetitle_high.png',dpi=300)
        plt.clf()


# %%
# effective update quotient Delta W x

def create_rms_quotient(df, layer):
    # Pre-compute the W0_op_norms dictionary for all widths that appear in the data
    unique_widths = df['width'].unique()

    # Create a pivot table to get required values for each (width, lr, warmup, step) combination
    pivot_df = df.pivot_table(
        index=['width', 'lr', 'warmup', 'step'],
        columns='statistic',
        values='value'
    )
    
    # Define a function to calculate the quotient for each row
    def op_quotient_fct(row):
        #op_quotient_fct = lambda x: (x[f'DeltaWx_{layer}'] / (x[f'DeltaW_op_{layer}'] * x[f'activation_norm_in_{layer}'])) #x['activation_norm_lnout'])) / np.log(x['width'])

        width = row.name[0]  # width is the first element of the MultiIndex
        if f'DeltaW_op_{layer}' in row and f'DeltaWx_{layer}' in row and f'activation_norm_in_{layer}' in row:
            return row[f'DeltaWx_{layer}'] / (row[f'DeltaW_frob_{layer}'] * row[f'activation_norm_in_{layer}'])
        return None
    
    # Apply the function to each row
    pivot_df[f'DeltaWx_rms_quotient_fannormalized_{layer}'] = pivot_df.apply(op_quotient_fct, axis=1)
    
    # Reset the index to convert back to regular DataFrame
    pivot_df = pivot_df.reset_index()
    
    # Create a new DataFrame with the derived statistic
    derived_rows = pivot_df[['width', 'lr', 'warmup', 'step', f'DeltaWx_rms_quotient_fannormalized_{layer}']].dropna(
        subset=[f'DeltaWx_rms_quotient_fannormalized_{layer}']
    ).copy()
    derived_rows['statistic'] = f'DeltaWx_rms_quotient_fannormalized_{layer}'
    derived_rows = derived_rows.rename(columns={f'DeltaWx_rms_quotient_fannormalized_{layer}': 'value'})
    
    # Append the new rows to the original DataFrame
    result_df = pd.concat([df, derived_rows], ignore_index=True)
    
    return result_df

df_result2 = create_rms_quotient(df,'out')
df_result2 = create_rms_quotient(df_result2,'lnout')
df_result2 = create_rms_quotient(df_result2,'mlp')


def create_opnorm_quotient(df, layer):
    # Pre-compute the W0_op_norms dictionary for all widths that appear in the data
    unique_widths = df['width'].unique()

    # Create a pivot table to get required values for each (width, lr, warmup, step) combination
    pivot_df = df.pivot_table(
        index=['width', 'lr', 'warmup', 'step'],
        columns='statistic',
        values='value'
    )
    
    # Define a function to calculate the quotient for each row
    def op_quotient_fct(row):
        #op_quotient_fct = lambda x: (x[f'DeltaWx_{layer}'] / (x[f'DeltaW_op_{layer}'] * x[f'activation_norm_in_{layer}'])) #x['activation_norm_lnout'])) / np.log(x['width'])

        width = row.name[0]  # width is the first element of the MultiIndex
        if f'DeltaW_op_{layer}' in row and f'DeltaWx_{layer}' in row and f'activation_norm_in_{layer}' in row:
            return row[f'DeltaWx_{layer}'] / (row[f'DeltaW_op_{layer}'] * row[f'activation_norm_in_{layer}'])
        return None
    
    # Apply the function to each row
    pivot_df[f'DeltaWx_op_quotient_{layer}'] = pivot_df.apply(op_quotient_fct, axis=1)
    
    # Reset the index to convert back to regular DataFrame
    pivot_df = pivot_df.reset_index()
    
    # Create a new DataFrame with the derived statistic
    derived_rows = pivot_df[['width', 'lr', 'warmup', 'step', f'DeltaWx_op_quotient_{layer}']].dropna(
        subset=[f'DeltaWx_op_quotient_{layer}']
    ).copy()
    derived_rows['statistic'] = f'DeltaWx_op_quotient_{layer}'
    derived_rows = derived_rows.rename(columns={f'DeltaWx_op_quotient_{layer}': 'value'})
    
    # Append the new rows to the original DataFrame
    result_df = pd.concat([df, derived_rows], ignore_index=True)
    
    return result_df

df_result = create_opnorm_quotient(df,'out')
df_result = create_opnorm_quotient(df_result,'lnout')
df_result = create_opnorm_quotient(df_result,'mlp')

plotter = WandBStatsPlotter(df_result)
plotter_rms = WandBStatsPlotter(df_result2)
ylog = True
for statname, ylim, title, layer, _ in zip(
    ['DeltaWx_rms_quotient_fannormalized_out','DeltaWx_op_quotient_out','DeltaWx_rms_quotient_fannormalized_lnout','DeltaWx_op_quotient_lnout'],
    [None,None,None,None],#[[0.49,1.01],[0.49,1.01],[-0.51,0.01],[-0.51,0.01]],
    [r'$||\Delta W_t x_t||/(||\Delta W_t||_{RMS}||x_t||)$',r'$||\Delta W_t x_t||/(||\Delta W_t||_{op}||x_t||)$',r'$||\Delta W_t x_t||/(||\Delta W_t||_{RMS}||x_t||)$',r'$||\Delta W_t x_t||/(||\Delta W_t||_{op}||x_t||)$'],
    ['out','out','lnout','lnout'],
    [r'RMS-alignment($\Delta W_t,x_t)$, outlayer', r'Operator-alignment($\Delta W_t,x_t)$, outlayer',r'RMS-alignment($\Delta W_t,x_t)$, last LN', r'Operator-alignment($\Delta W_t,x_t)$, last LN']):
    if layer == 'out':
        if ylog:
            ylim= [0.09,1.01]
        else:
            ylim = [-0.01,1.01]
    else:
        ylim = [9.99e-4,2.01]
    if '_rms_' in statname:
        plt_deltaWx = plotter_rms.plot_over_time(
            statistic=statname, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,height=1.2) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statname}_{base_lr}_{lrexpon}_{exp_short}_mainfig_test_largefont.png',dpi=300)
        plt.clf()
    else:
        plt_deltaWx = plotter.plot_over_time(
            statistic=statname, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,height=1.2) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statname}_{base_lr}_{lrexpon}_{exp_short}_mainfig_test_largefont_high.png',dpi=300)
        plt.clf()

if 'standard' in project_name:
    expons = ('-0.26', '-0.0006','-0.065') #r'$-6.1 \cdot 10^{-4}$'
    text_pos = ((1600,0.07),(1150,0.55),(1350,0.13))
else:
    expons, text_pos = None,None

for statnames, ylim, title, _ in zip(
    [['DeltaWx_rms_quotient_fannormalized_lnout','DeltaWx_rms_quotient_fannormalized_mlp','DeltaWx_rms_quotient_fannormalized_out'],['DeltaWx_op_quotient_lnout','DeltaWx_op_quotient_mlp','DeltaWx_op_quotient_out']],
    [None,None,None,None],#[[0.49,1.01],[0.49,1.01],[-0.51,0.01],[-0.51,0.01]],
    [r'$||\Delta W_t x_t||/(||\Delta W_t||_{RMS}||x_t||)$',r'$||\Delta W_t x_t||/(||\Delta W_t||_{op}||x_t||)$'],
    [r'RMS-alignment($\Delta W_t,x_t)$', r'Operator-alignment($\Delta W_t,x_t)$']):
    # if layer == 'out':
    #     if ylog:
    #         ylim= [0.09,1.01]
    #     else:
    #         ylim = [-0.01,1.01]
    # else:
    #     ylim = [9.99e-4,2.01]
    if '_op_' in statnames[0]:
        legend_title = 'width'
        legend_labels = [256,1024,4096]
        ylim = [0.03,1.01]
    else:
        legend_title = 'layer'
        legend_labels = ['last LN','mlp hidden','outlayer']
        #ylim = [0.099,1.01]
    if '_rms_' in statnames[0]:
        plt_deltaWx = plotter_rms.plot_over_time(
            statistic=statnames, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,legend_title=legend_title,legend_labels=legend_labels,height=1.2, title_font=14) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statnames[0]}_{base_lr}_{lrexpon}_{exp_short}_mainfig_multilayer_test_largefont_largetitle_high.png',dpi=300)
        plt.clf()
    else:
        plt_deltaWx = plotter.plot_over_time(
            statistic=statnames, widths=widths_to_analyze, lr_by_width=lr_by_width, log_x=True, log_y=ylog, ylim=ylim, y_label = None,title=title,legend_title=legend_title,legend_labels=legend_labels,height=1.2,expons=expons,text_pos=text_pos, title_font=14) #, SP, LR={base_lr}*(width/256)**{lrexpon}')
        plt_deltaWx.savefig(plot_path+f'{statnames[0]}_{base_lr}_{lrexpon}_{exp_short}_mainfig_multilayer_test_ylim_largefont_largetitle_annot_high.png',dpi=300)
        plt.clf()

# %%
# get the width-scaling exponents at the end of training
from scipy import stats

def exponent(x,y):
    # x: (x_1, x_2), y: (y_1, y_2)
    # assuming y = c x^d, determine d
    return (np.log(y[0])-np.log(y[-1]))/(np.log(x[0])-np.log(x[-1]))


def get_expon(xs,ys):
    x_log,y_log = np.log10(xs), np.log10(ys)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
    return slope, intercept

widths = [256,4096]
step = 2002
layer = 'out'

width=widths[0]
W0op = df[(df['width'] == width) & (df['statistic'] == f'W_opnorm_{layer}') &  (df['step'] == 0) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
WDeltax, Deltax= df[(df['width'] == width) & (df['statistic'] == f'WDeltax_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0],df[(df['width'] == width) & (df['statistic'] == f'Deltax_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0]
base_stat = WDeltax / (W0op * Deltax)

width = widths[-1]
W0op = df[(df['width'] == width) & (df['statistic'] == f'W_opnorm_{layer}') &  (df['step'] == 0) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
WDeltax, Deltax= df[(df['width'] == width) & (df['statistic'] == f'WDeltax_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0],df[(df['width'] == width) & (df['statistic'] == f'Deltax_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0]
wide_stat = WDeltax / (W0op * Deltax)

print('WDeltax operator', layer, base_stat,wide_stat, exponent([256,4096],[base_stat,wide_stat]), get_expon(widths,[base_stat,wide_stat])[0])

width=widths[0]
#W0op = df[(df['width'] == width) & (df['statistic'] == f'W_opnorm_{layer}') &  (df['step'] == 0) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
DeltaWop = df[(df['width'] == width) & (df['statistic'] == f'DeltaW_op_{layer}') &  (df['step'] == step) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
DeltaWx, xt= df[(df['width'] == width) & (df['statistic'] == f'DeltaWx_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0],df[(df['width'] == width) & (df['statistic'] == f'activation_norm_in_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0]
base_stat = DeltaWx / (DeltaWop * xt)

width = widths[-1]
#W0op = df[(df['width'] == width) & (df['statistic'] == f'W_opnorm_{layer}') &  (df['step'] == 0) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
DeltaWop = df[(df['width'] == width) & (df['statistic'] == f'DeltaW_op_{layer}') &  (df['step'] == step) & (df['lr']==lr_by_width[width])]['value'].iloc[0]
DeltaWx, xt= df[(df['width'] == width) & (df['statistic'] == f'DeltaWx_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0],df[(df['width'] == width) & (df['statistic'] == f'activation_norm_in_{layer}') &  (df['step'] == step)& (df['lr']==lr_by_width[width])]['value'].iloc[0]
wide_stat = DeltaWx / (DeltaWop * xt)

print('DeltaWx operator', layer,base_stat,wide_stat, exponent([256,4096],[base_stat,wide_stat]),get_expon(widths,[base_stat,wide_stat])[0])

# %%