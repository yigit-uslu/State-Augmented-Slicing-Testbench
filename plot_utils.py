from collections import defaultdict
from itertools import cycle
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from core.Slice import Slice
import re

# plt.rcParams.update({'font.size': 8})
plt.rcParams['patch.edgecolor'] = 'none'
# plt.rcParams['figure.figsize'] = (18, 6)
FIG_ROWSIZE = 6
FIG_COLSIZE = 6
FIG_FONTSIZE = 8
FIG_PADDING = 7


def create_color_cyclers(n = 5):
    colorList = ['blue', 'red', 'orange', 'green', 'magenta']
    colorcycle = cycle(colorList[:n])
    return colorcycle

def create_marker_cyclers(n = 5):
    markerList = ['o', '+', 'x', '*', '.']
    markercycle = cycle(markerList[:n])
    return markercycle


def plot_train_primal_evolution_over_epochs(train_state_over_epochs, save_path = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)
    fig_padding = kwargs.get('fig_padding', FIG_PADDING)

    plt.rcParams.update({'font.size': fig_fontSize})

    fig, axs = plt.subplots(1, 2, figsize = (2 * fig_colSize, 1 * fig_rowSize))

    pgrad_norm_arr = []
    Lagrangian_arr = []
    for train_state in train_state_over_epochs:
        pgrad_norm_arr.append(train_state.primal_optim_state.pgrad_norm)
        Lagrangian_arr.append(train_state.primal_optim_state.Lagrangian)
    pgrad_norm_arr = np.array(pgrad_norm_arr)
    Lagrangian_arr = np.array(Lagrangian_arr)

    axs[0].semilogy(np.arange(pgrad_norm_arr.shape[0]), pgrad_norm_arr, '-dr', label = ['pgrad norm'])
    axs[1].plot(np.arange(Lagrangian_arr.shape[0]), Lagrangian_arr, '-dr', label = ['Lagrangian'])
    axs[0].set_ylabel('pgrad norm')
    axs[1].set_ylabel('Lagrangian')

    for ax in axs.flat:
        ax.grid(True)
        ax.set_xlabel('Iteration (k)')
        ax.legend()

    fig.suptitle('Train primal optim. evolution over epochs')
    fig.tight_layout(pad=fig_padding)
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/train_state_evolution.png', dpi = 300)
    plt.close()



def plot_test_evolution_over_epochs(self, test_states_over_epochs, test_config_names, save_path = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)
    fig_padding = kwargs.get('fig_padding', FIG_PADDING)
    q = kwargs.get('metric_quantile', 0.95)
    plot_actual_metrics = kwargs.get('plot_actual_metrics', True)

    plt.rcParams.update({'font.size': fig_fontSize})

    n_obj = 1
    n_samples, n_constraints = test_states_over_epochs[0][0].metrics.constraints.shape
    
    if n_samples == 1:
        n_cols = 1
    else:
        n_cols = 3

    n_metrics = len(test_states_over_epochs)
    
    plt.figure(1) # obj test evolution
    fig_1, axs_1 = plt.subplots(n_obj, n_cols, figsize = (n_cols * fig_colSize, n_obj * fig_rowSize))

    fig_1_colorcycler = create_color_cyclers(n = n_metrics)
    fig_1_markercycler = create_marker_cyclers(n = n_metrics)


    plt.figure(2) # constraint metrics test evolution
    fig_2, axs_2 = plt.subplots(n_constraints, n_cols, figsize = (n_cols * fig_colSize, n_constraints * fig_rowSize))

    fig_2_colorcycler = create_color_cyclers(n = n_metrics)
    fig_2_markercycler = create_marker_cyclers(n = n_metrics)

    # plt.figure(3) # throughput test evolution
    # fig_3, axs_3 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list()) * fig_colSize, 1 * fig_rowSize))

    # plt.figure(4) # rate test evolution
    # fig_4, axs_4 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list()) * fig_colSize, 1 * fig_rowSize))

    # plt.figure(5) # latency test evolution
    # fig_5, axs_5 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list()) * fig_colSize, 1 * fig_rowSize))


    for test_state_over_epochs, test_config_name in zip(test_states_over_epochs, test_config_names):
        obj_mean_arr = []
        obj_max_arr = []
        obj_percentile_arr = []
        constraints_mean_arr = []
        constraints_max_arr = []
        constraints_percentile_arr = []
        for test_state in test_state_over_epochs:
            obj_mean_arr.append(np.mean(test_state.metrics.obj, axis = 0))
            obj_max_arr.append(np.max(test_state.metrics.obj, axis = 0))
            obj_percentile_arr.append(np.quantile(test_state.metrics.obj, q = q, axis = 0))
            constraints_mean_arr.append(np.mean(test_state.metrics.constraints, axis = 0))
            constraints_max_arr.append(np.max(test_state.metrics.constraints, axis = 0))
            constraints_percentile_arr.append(np.quantile(test_state.metrics.constraints, q = q, axis = 0))

        obj_mean_arr = np.array(obj_mean_arr)
        obj_max_arr = np.array(obj_max_arr)
        obj_percentile_arr = np.array(obj_percentile_arr)

        constraints_mean_arr = np.array(constraints_mean_arr)
        constraints_max_arr = np.array(constraints_max_arr)
        constraints_percentile_arr = np.array(constraints_percentile_arr)

        if plot_actual_metrics:
            obj_mean_arr = self.obj_fnc.to_metric(obj_mean_arr)
            obj_max_arr = self.obj_fnc.to_metric(obj_max_arr)
            obj_percentile_arr = self.obj_fnc.to_metric(obj_percentile_arr)

            for i, constraint_fnc in enumerate(self.constraint_fncs):
                constraints_mean_arr[:, i] = constraint_fnc.to_metric(constraints_mean_arr[:, i])
                constraints_max_arr[:, i] = constraint_fnc.to_metric(constraints_max_arr[:, i])
                constraints_percentile_arr[:, i] = constraint_fnc.to_metric(constraints_percentile_arr[:, i])

        n_epochs = obj_mean_arr.shape

        plt.figure(fig_1)
        marker = next(fig_1_markercycler)
        color = next(fig_1_colorcycler)

        if n_cols == 1:
            axs_1.plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr, '-', marker = marker, color = color, label = [f'{test_config_name}'])
        else:
            axs_1[0].plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr, '-', marker = marker, color = color, label = [f'{test_config_name}'])
            axs_1[1].plot(np.arange(obj_max_arr.shape[0]), obj_max_arr, '-', marker = marker, color = color, label = [f'{test_config_name}'])
            axs_1[2].plot(np.arange(obj_percentile_arr.shape[0]), obj_percentile_arr, '-', marker = marker, color = color, label = [f'{test_config_name}'])
        

        plt.figure(fig_2)
        marker = next(fig_2_markercycler)
        color = next(fig_2_colorcycler)

        for i in range(n_constraints):
            if n_cols == 1:
                axs_2[i].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, i], '-', marker = marker, color = color, label = [f'{test_config_name}'])
            else:
                axs_2[i, 0].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, i], '-', marker = marker, color = color, label = [f'{test_config_name}'])
                axs_2[i, 1].plot(np.arange(constraints_max_arr.shape[0]), constraints_max_arr[:, i], '-', marker = marker, color = color, label = [f'{test_config_name}'])
                axs_2[i, 2].plot(np.arange(constraints_percentile_arr.shape[0]), constraints_percentile_arr[:, i], '-', marker = marker, color = color, label = [f'{test_config_name}'])


    plt.figure(fig_1)
    if n_cols == 1:
        axs_1.set_ylabel('Obj (bps/Hz)')
        axs_1.grid(True)
        axs_1.set_xlabel('Iteration (k)')
        axs_1.legend()
    else:
        if plot_actual_metrics:
            axs_1[0].set_ylabel('Obj (bps/Hz) (mean)')
            axs_1[1].set_ylabel('Obj (bps/Hz) (min)')
            axs_1[2].set_ylabel(f'Obj (bps/Hz) ({q} - percentile)')

        else:
            axs_1[0].set_ylabel('Obj (mean)')
            axs_1[1].set_ylabel('Obj (max)')
            axs_1[2].set_ylabel(f'Obj ({q} - percentile)')

        for ax in axs_1.flat:
            ax.grid(True)
            ax.set_xlabel('Iteration (k)')
            # ax.set_xticks(range(self.config.n_primal_iters, (n_epochs + 1) * self.config.n_primal_iters, self.config.n_primal_iters))
            ax.legend()

    fig_1.suptitle(f'Test evolution obj mean/max/percentile \n across {n_samples} slicing networks')
    fig_1.tight_layout(pad=fig_padding)
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/test_evolution_obj.png', dpi = 300)
    plt.close(fig_1)

    plt.figure(fig_2)
    for i in range(n_constraints):
        if plot_actual_metrics:
            if n_cols == 1:
                axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__}')
                axs_2[i].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
            else:
                axs_2[i, 0].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (mean)')
                axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (max)')
                axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} ({q} - quantile)')

                axs_2[i, 0].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                axs_2[i, 1].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                axs_2[i, 2].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')

        else:
            if n_cols == 1:
                axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} slack')
                
            else:
                axs_2[i, 0].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (mean)')
                axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (max)')
                axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} slack ({q} - quantile)')

            # axs_2[i, 0].axhline(y = 0, xmin = 0, xmax = 1, linestyle = '--', color = 'k')
            # axs_2[i, 1].axhline(y = 0, xmin = 0, xmax = 1, linestyle = '--', color = 'k')
            # axs_2[i, 2].axhline(y = 0, xmin = 0, xmax = 1, linestyle = '--', color = 'k')

    for ax in axs_2.flat:
        ax.grid(True)
        ax.set_xlabel('Iteration (k)')
        # ax.set_xticks(range(self.config.n_primal_iters, (n_epochs + 1) * self.config.n_primal_iters, self.config.n_primal_iters))
        ax.legend()

    if plot_actual_metrics:
        fig_2.suptitle(f'Test evolution constraint metrics mean/max/percentile \n across {n_samples} slicing networks')
    else:
        fig_2.suptitle(f'Test evolution constraint slacks mean/max/percentile \n across {n_samples} slicing networks')
    fig_2.tight_layout(pad=fig_padding)
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    if plot_actual_metrics:
        plt.savefig(f'{save_path}/test_evolution_constraint_metrics.png', dpi = 300)
    else:
        plt.savefig(f'{save_path}/test_evolution_constraint_slacks.png', dpi = 300)
    plt.close(fig_2)
    # plt.close(2)



def plot_test_evolution_over_slices(self, test_metrics_over_slices, test_config_names, save_path = None, network_idx = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)
    fig_padding = kwargs.get('fig_padding', FIG_PADDING)
    subfig_expand_factor = 1.3
    q = kwargs.get('metric_quantile', 0.95)
    plot_actual_metrics = kwargs.get('plot_actual_metrics', True)
    save_subplots = kwargs.get('save_subplots', True)

    plt.rcParams.update({'font.size': fig_fontSize})

    n_obj = 1
    n_samples, n_constraints, T_slices = test_metrics_over_slices[0][0].metrics_over_slices.constraints_over_slices.shape
    n_slices = test_metrics_over_slices[0][0].metrics_over_slices.Ps_over_slices.shape[1]

    n_epochs = len(test_metrics_over_slices[0])
    n_metrics = len(test_metrics_over_slices)

    if network_idx is None:
        network_idx = range(n_samples)

    all_obj_mean_arr = []
    all_obj_max_arr = []
    all_obj_percentile_arr = []
    all_constraints_mean_arr = []
    all_constraints_max_arr = []
    all_constraints_percentile_arr = []
    all_lambdas_arr = []
    all_Ps_arr = []
    all_constraint_violations_arr = []

    all_throughputs_arr = []
    all_rates_arr = []
    all_latencies_quantile_arr = []
    all_latencies_avg_arr = []
    all_slice_associations_arr = []

    for test_metric_over_slices in test_metrics_over_slices:

        obj_mean_arr = [test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        obj_max_arr = [test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1).max(0) for t in range(T_slices)]
        obj_percentile_arr = [np.quantile(test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1), q=q, axis = 0) for t in range(T_slices)]
        constraints_mean_arr = [test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        constraints_max_arr = [test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1).max(0) for t in range(T_slices)]
        constraints_percentile_arr = [np.quantile(test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1), q=q, axis = 0) for t in range(T_slices)]
        Ps_arr = test_metric_over_slices[-1].metrics_over_slices.Ps_over_slices
        lambdas_arr = test_metric_over_slices[-1].metrics_over_slices.lambdas_over_slices
        # constraint_violations_arr = [test_metric_over_slices[-1].metrics_over_slices.constraint_violations_over_slices[:, :, :, max(0, t-self.config.T_slices['train']):(t+1) ].mean(-1) for t in range(T_slices)]
        constraint_violations_arr = [test_metric_over_slices[-1].metrics_over_slices.constraint_violations_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]

        throughputs_arr = [test_metric_over_slices[-1].metrics_over_slices.slicing_metrics_over_slices['throughput'][:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        rates_arr = [test_metric_over_slices[-1].metrics_over_slices.slicing_metrics_over_slices['rate'][:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        latencies_quantile_arr = [test_metric_over_slices[-1].metrics_over_slices.slicing_metrics_over_slices['latency_quantile'][:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        latencies_avg_arr = [test_metric_over_slices[-1].metrics_over_slices.slicing_metrics_over_slices['latency_avg'][:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        slice_associations_arr = test_metric_over_slices[-1].metrics_over_slices.slicing_metrics_over_slices['slice_association']

        obj_mean_arr = np.array(obj_mean_arr)
        obj_max_arr = np.array(obj_max_arr)
        obj_percentile_arr = np.array(obj_percentile_arr)
        constraints_mean_arr = np.array(constraints_mean_arr)
        constraints_max_arr = np.array(constraints_max_arr)
        constraints_percentile_arr = np.array(constraints_percentile_arr)
        constraint_violations_arr = np.array(constraint_violations_arr)
        # obj_mean_arr = np.moveaxis(np.array(obj_mean_arr), source=-1, destination=0)
        # obj_max_arr = np.moveaxis(np.array(obj_max_arr), source=-1, destination=0)
        # obj_percentile_arr = np.moveaxis(np.array(obj_percentile_arr), source=-1, destination=0)
        # constraints_mean_arr = np.moveaxis(np.array(constraints_mean_arr), source=-1, destination=0)
        # constraints_max_arr = np.moveaxis(np.array(constraints_max_arr), source=-1, destination=0)
        # constraints_percentile_arr = np.moveaxis(np.array(constraints_percentile_arr), source=-1, destination=0)
        Ps_arr = np.moveaxis(np.array(Ps_arr), source=-1, destination=0)
        lambdas_arr = np.moveaxis(np.array(lambdas_arr), source=-1, destination=0)

        slice_associations_arr = np.moveaxis(np.array(slice_associations_arr), source=-1, destination = 0)

        throughputs_arr = np.array(throughputs_arr) / self.config.BW_total # bps/Hz
        rates_arr = np.array(rates_arr) / self.config.BW_total # bps/Hz
        latencies_quantile_arr = np.array(latencies_quantile_arr) * 1000 # ms
        latencies_avg_arr = np.array(latencies_avg_arr) * 1000 # ms

        all_obj_mean_arr.append(obj_mean_arr)
        all_obj_max_arr.append(obj_max_arr)
        all_obj_percentile_arr.append(obj_percentile_arr)
        all_constraints_mean_arr.append(constraints_mean_arr)
        all_constraints_max_arr.append(constraints_max_arr)
        all_constraints_percentile_arr.append(constraints_percentile_arr)
        all_Ps_arr.append(Ps_arr)
        all_lambdas_arr.append(lambdas_arr)
        all_constraint_violations_arr.append(constraint_violations_arr)

        all_throughputs_arr.append(throughputs_arr)
        all_rates_arr.append(rates_arr)
        all_latencies_avg_arr.append(latencies_avg_arr)
        all_latencies_quantile_arr.append(latencies_quantile_arr)
        all_slice_associations_arr.append(slice_associations_arr)

    
    for network_id in network_idx:
        
        if n_samples == 1:
            n_cols = 1
        else:
            n_cols = 3
    
        plt.figure(1) # obj
        fig_1, axs_1 = plt.subplots(n_obj, n_cols, figsize = (n_cols * fig_colSize, n_obj * fig_rowSize))
        fig_1.tight_layout(pad=fig_padding)

        plt.figure(2) # constraint slacks
        fig_2, axs_2 = plt.subplots(n_constraints, n_cols, figsize = (n_cols * fig_colSize, n_constraints * fig_rowSize))
        fig_2.tight_layout(pad=fig_padding)

        plt.figure(3) # Ps
        fig_3, axs_3 = plt.subplots(1, n_slices, figsize = (n_slices * fig_colSize, 1 * fig_rowSize))
        fig_3.tight_layout(pad=fig_padding)

        plt.figure(4) # dual multipliers
        fig_4, axs_4 = plt.subplots(n_constraints, 1, figsize = (1 * fig_colSize, n_constraints * fig_rowSize))
        fig_4.tight_layout(pad=fig_padding)

        plt.figure(5) # individual constraint violations 
        fig_5, axs_5 = plt.subplots(n_metrics, n_constraints, figsize = (n_constraints * fig_colSize, n_metrics * fig_rowSize), squeeze=False)
        fig_5.tight_layout(pad=fig_padding)
        
        plt.figure(6) # throughput test evolution
        fig_6, axs_6 = plt.subplots(n_metrics, 3, figsize = (3 * fig_colSize, n_metrics * fig_rowSize))
        fig_6.tight_layout(pad=fig_padding)

        plt.figure(7) # rate test evolution
        fig_7, axs_7 = plt.subplots(n_metrics, 3, figsize = (3 * fig_colSize, n_metrics * fig_rowSize))
        fig_7.tight_layout(pad=fig_padding)

        plt.figure(8) # latency quantile test evolution
        fig_8, axs_8 = plt.subplots(n_metrics, 3, figsize = (3 * fig_colSize, n_metrics * fig_rowSize))
        fig_8.tight_layout(pad=fig_padding)

        plt.figure(9) # latency avg test evolution
        fig_9, axs_9 = plt.subplots(n_metrics, 3, figsize = (3 * fig_colSize, n_metrics * fig_rowSize))
        fig_9.tight_layout(pad=fig_padding)

        plt.figure(10) # slice associations test evolution
        fig_10, axs_10 = plt.subplots(n_metrics, 3, figsize = (3 * fig_colSize, n_metrics * fig_rowSize))
        fig_10.tight_layout(pad=fig_padding)

        # plt.figure(6) # throughput test evolution
        # fig_6, axs_6 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list().pop(Slice.IA)) * fig_colSize, 1 * fig_rowSize))

        # plt.figure(7) # rate test evolution
        # fig_7, axs_7 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list().pop(Slice.IA)) * fig_colSize, 1 * fig_rowSize))

        # plt.figure(8) # latency quantile test evolution
        # fig_8, axs_8 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list().pop(Slice.IA)) * fig_colSize, 1 * fig_rowSize))

        # plt.figure(9) # latency avg test evolution
        # fig_9, axs_9 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list().pop(Slice.IA)) * fig_colSize, 1 * fig_rowSize))

        # plt.figure(10) # slice associations test evolution
        # fig_10, axs_10 = plt.subplots(1, len(Slice.list()), figsize = (len(Slice.list().pop(Slice.IA)) * fig_colSize, 1 * fig_rowSize))


        for test_config_id, test_config_name in enumerate(test_config_names):
            obj_mean_arr = all_obj_mean_arr[test_config_id]
            obj_max_arr = all_obj_max_arr[test_config_id]
            obj_percentile_arr = all_obj_percentile_arr[test_config_id]
            constraints_mean_arr = all_constraints_mean_arr[test_config_id]
            constraints_max_arr = all_constraints_max_arr[test_config_id]
            constraints_percentile_arr = all_constraints_percentile_arr[test_config_id]
            lambdas_arr = all_lambdas_arr[test_config_id]
            Ps_arr = all_Ps_arr[test_config_id]
            constraint_violations_arr = all_constraint_violations_arr[test_config_id]

            throughputs_arr = all_throughputs_arr[test_config_id]
            rates_arr = all_rates_arr[test_config_id]
            latencies_quantile_arr = all_latencies_quantile_arr[test_config_id]
            latencies_avg_arr = all_latencies_avg_arr[test_config_id]
            slice_associations_arr = all_slice_associations_arr[test_config_id]
            
            for logger in self.loggers:
                logger(constraint_violations_arr)

            if plot_actual_metrics:
                obj_mean_arr[:, network_id] = self.obj_fnc.to_metric(obj_mean_arr[:, network_id])
                obj_max_arr = self.obj_fnc.to_metric(obj_max_arr)
                obj_percentile_arr = self.obj_fnc.to_metric(obj_percentile_arr)

                for i, constraint_fnc in enumerate(self.constraint_fncs):
                    constraints_mean_arr[:, network_id, i] = constraint_fnc.to_metric(constraints_mean_arr[:, network_id, i])
                    constraints_max_arr[:, i] = constraint_fnc.to_metric(constraints_max_arr[:, i])
                    constraints_percentile_arr[:, i] = constraint_fnc.to_metric(constraints_percentile_arr[:, i])

            plt.figure(fig_1)
            if n_cols == 1:
                axs_1.plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr[:, network_id], '-', label = f'{test_config_name}')
            else:
                axs_1[0].plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr[:, network_id], '-', label = f'{test_config_name}')
                axs_1[1].plot(np.arange(obj_max_arr.shape[0]), obj_max_arr, '-', label = f'{test_config_name}')
                axs_1[2].plot(np.arange(obj_percentile_arr.shape[0]), obj_percentile_arr, '-', label = f'{test_config_name}')
            
        
            plt.figure(fig_2)
            for i in range(n_constraints):
                if n_cols == 1:
                    axs_2[i].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, network_id, i], '-', label = f'{test_config_name}')
                else:
                    axs_2[i, 0].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, network_id, i], '-', label = f'{test_config_name}')
                    axs_2[i, 1].plot(np.arange(constraints_max_arr.shape[0]), constraints_max_arr[:, i], '-', label = f'{test_config_name}')
                    axs_2[i, 2].plot(np.arange(constraints_percentile_arr.shape[0]), constraints_percentile_arr[:, i], '-', label = f'{test_config_name}')

            plt.figure(fig_3)
            for i in range(n_slices):
                axs_3[i].plot(np.arange(Ps_arr.shape[0]), Ps_arr[:, network_id, i], '-', label = f'{test_config_name}')

            if test_config_name in ['state-augmented-slicing']:
                plt.figure(fig_4)
                for i in range(n_constraints):
                    axs_4[i].plot(np.arange(lambdas_arr.shape[0]), lambdas_arr[:, network_id, i], '-', label = f'{test_config_name}')

            # if test_config_name in ['state-augmented-slicing']:
            plt.figure(fig_5)
            for i, ax in enumerate(axs_5[test_config_id].flat):
                axs_5[test_config_id, i].plot(np.arange(constraint_violations_arr.shape[0]), constraint_violations_arr[:, network_id, i], '-', label = f'{test_config_name}')
            # for i in range(n_constraints):
            #     axs_5[i].plot(np.arange(constraint_violations_arr.shape[0]), constraint_violations_arr[:, network_id, i], '-', label = f'{test_config_name}')
      

            plt.figure(fig_6)
            for i, ax in enumerate(axs_6[test_config_id].flat):
                slice_idx = slice_associations_arr[0, network_id] == Slice.list()[i].value
                ax.plot(np.arange(throughputs_arr.shape[0]), throughputs_arr[:, network_id, slice_idx], '-', label = [f'Client #{_}' for _ in np.where(slice_idx)[0].tolist()])

            # if test_config_name in ['state-augmented-slicing']:
            plt.figure(fig_7)
            for i, ax in enumerate(axs_7[test_config_id].flat):
                slice_idx = slice_associations_arr[0, network_id] == Slice.list()[i].value
                ax.plot(np.arange(rates_arr.shape[0]), rates_arr[:, network_id, slice_idx], '-', label = [f'Client #{_}' for _ in np.where(slice_idx)[0].tolist()])

            # if test_config_name in ['state-augmented-slicing']:
            plt.figure(fig_8)
            for i, ax in enumerate(axs_8[test_config_id].flat):
                slice_idx = slice_associations_arr[0, network_id] == Slice.list()[i].value
                ax.plot(np.arange(latencies_quantile_arr.shape[0]), latencies_quantile_arr[:, network_id, slice_idx], '-', label = [f'Client #{_}' for _ in np.where(slice_idx)[0].tolist()])

            # if test_config_name in ['state-augmented-slicing']:
            plt.figure(fig_9)
            for i, ax in enumerate(axs_9[test_config_id].flat):
                slice_idx = slice_associations_arr[0, network_id] == Slice.list()[i].value
                ax.plot(np.arange(latencies_avg_arr.shape[0]), latencies_avg_arr[:, network_id, slice_idx], '-', label = [f'Client #{_}' for _ in np.where(slice_idx)[0].tolist()])

            # if test_config_name in ['state-augmented-slicing']:
            plt.figure(fig_10)
            for i, ax in enumerate(axs_10[test_config_id].flat):
                slice_idx = slice_associations_arr[0, network_id] == Slice.list()[i].value
                ax.plot(np.arange(slice_associations_arr.shape[0]), slice_associations_arr[:, network_id, slice_idx], '-', label = [f'Client #{_}' for _ in np.where(slice_idx)[0].tolist()])

        
        plt.figure(fig_1)
        if n_cols == 1:
            axs_1.set_ylabel('Obj (bps/Hz)')
            axs_1.grid(True)
            axs_1.set_xlabel('Slice window (t)')
            axs_1.legend()
        else:
            axs_1[0].set_ylabel('Obj (this network)')
            axs_1[1].set_ylabel('Obj (worst network)')
            axs_1[2].set_ylabel(f'Obj ({100 * q}-percentile network)')

            for ax in axs_1.flat:
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()

        fig_1.suptitle(f'Test evolution obj mean/max/percentile \n network #{network_id}')
        # fig_1.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_obj.png', dpi = 300)
        plt.close(fig_1)

        plt.figure(fig_2)
        for i in range(n_constraints):
            if plot_actual_metrics:
                if n_cols == 1:
                    axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__}')
                    axs_2[i].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')

                else:
                    axs_2[i,0].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (this network)')
                    axs_2[i,1].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (worst network)')
                    axs_2[i,2].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} ({100*q}-percentile network)')
                    axs_2[i, 0].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                    axs_2[i, 1].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                    axs_2[i, 2].axhline(y = self.constraint_fncs[i].to_metric.tol, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
            else:
                if n_cols == 1:
                    axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} slack')
                else:
                    axs_2[i, 0].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (this network)')
                    # axs_2[i, 0].axhline(y = 0., xmin = 0, xmax = 1, linestyle = '--', color = 'k')
                    axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (worst network)')
                    axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} slack ({100*q}-percentile network)')

        for ax in axs_2.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        if plot_actual_metrics:
            fig_2.suptitle(f'Test evolution constraint metrics mean/max/percentile \n network #{network_id}')
        else:
            fig_2.suptitle(f'Test evolution constraint slacks mean/max/percentile \n network #{network_id}')
        # fig_2.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        if plot_actual_metrics:
            plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_metrics.png', dpi = 300)
        else:
            plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_slacks.png', dpi = 300)
        plt.close(fig_2)


        plt.figure(fig_3)
        for i in range(n_slices):
            axs_3[i].set_ylabel(f'{Slice.list()[i]}')

        for ax in axs_3.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_3.suptitle(f'Test evolution of resource allocs. \n network #{network_id}')
        # fig_3.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_Ps.png', dpi = 300)
        plt.close(fig_3)


        plt.figure(fig_4)
        for i in range(n_constraints):
            axs_4[i].set_ylabel(f'{self.constraint_fncs[i].__name__}')

        for ax in axs_4.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_4.suptitle(f'Test evolution of dual multipliers \n network #{network_id}')
        # fig_4.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_lambdas.png', dpi = 300)
        plt.close(fig_4)


        plt.figure(fig_5)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_5[i].flat):
                ax.set_ylabel(f'{self.constraint_fncs[j].__name__}')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')

                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_5.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/constraint_violations/', exist_ok=True)
                    fig_5.savefig(f'{save_path}/network_{network_id}/constraint_violations/epoch_{n_epochs}_test_evolution_constraint_violations_{test_config_names[i]}_{self.constraint_fncs[j].__name__}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))

        fig_5.suptitle(f'Test evolution of average constraint violation indicators \n network #{network_id}')
        # fig_5.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_violations.png', dpi = 300)
        plt.close(fig_5)


        plt.figure(fig_6)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_6[i].flat):
                ax.axhline(y = self.config.r_min, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                ax.set_ylabel('Throughput (bps/Hz)')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')
                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_6.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/throughputs/', exist_ok=True)
                    fig_6.savefig(f'{save_path}/network_{network_id}/throughputs/epoch_{n_epochs}_test_evolution_throughputs_{test_config_names[i]}_{Slice.list()[j]}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))


        fig_6.suptitle(f'Test evolution of average throughputs \n network #{network_id}')
        # fig_6.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_throughputs.png', dpi = 300)
        plt.close(fig_6)


        plt.figure(fig_7)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_7[i].flat):
                ax.axhline(y = self.config.r_min, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                ax.set_ylabel('Rate (bps/Hz)')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')

                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_7.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/rates/', exist_ok=True)
                    fig_7.savefig(f'{save_path}/network_{network_id}/rates/epoch_{n_epochs}_test_evolution_rates_{test_config_names[i]}_{Slice.list()[j]}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))


        fig_7.suptitle(f'Test evolution of average rates \n network #{network_id}')
        # fig_7.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_rates.png', dpi = 300)
        plt.close(fig_7)


        plt.figure(fig_8)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_8[i].flat):
                ax.axhline(y = 1000 * self.config.l_max, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                ax.set_ylabel('Quantile-latency (ms)')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')

                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_8.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/latencies_quantile/', exist_ok=True)
                    fig_8.savefig(f'{save_path}/network_{network_id}/latencies_quantile/epoch_{n_epochs}_test_evolution_latencies_quantile_{test_config_names[i]}_{Slice.list()[j]}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))


        fig_8.suptitle(f'Test evolution of average quantile-latencies \n network #{network_id}')
        # fig_8.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_latencies_quantile.png', dpi = 300)
        plt.close(fig_8)


        plt.figure(fig_9)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_9[i].flat):
                ax.axhline(y = 1000 * self.config.l_max, xmin = 0, xmax = 1, linestyle = '--', color = 'r')
                ax.set_ylabel('Avg-latency (ms)')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')

                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_9.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/latencies_avg/', exist_ok=True)
                    fig_9.savefig(f'{save_path}/network_{network_id}/latencies_avg/epoch_{n_epochs}_test_evolution_latencies_avg_{test_config_names[i]}_{Slice.list()[j]}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))


        fig_9.suptitle(f'Test evolution of average avg-latencies \n network #{network_id}')
        # fig_9.tight_layout(pad=fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_latencies_avg.png', dpi = 300)
        plt.close(fig_9)


        plt.figure(fig_10)
        for i in range(n_metrics):
            for j, ax in enumerate(axs_10[i].flat):
                ax.set_ylabel('Slice associations')
                ax.grid(True)
                ax.set_xlabel('Slice window (t)')
                ax.legend()
                ax.set_title(f'{test_config_names[i]}')

                if save_subplots:
                    # Save subplot as a separate figure
                    extent = ax.get_window_extent().transformed(fig_10.dpi_scale_trans.inverted())
                    os.makedirs(f'{save_path}/network_{network_id}/slice_associations/', exist_ok=True)
                    fig_10.savefig(f'{save_path}/network_{network_id}/slice_associations/epoch_{n_epochs}_test_evolution_slice_associations_{test_config_names[i]}_{Slice.list()[j]}.png', dpi = 300, bbox_inches=extent.expanded(subfig_expand_factor, subfig_expand_factor))

        fig_10.suptitle(f'Test evolution of slice associations \n network #{network_id}')
        # fig_10.tight_layout(pad = fig_padding)
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_slice_associations.png', dpi = 300)
        plt.close(fig_10)
    

        # for i, test_metric_over_slices in enumerate(test_metrics_over_slices):
        #     obj_mean_arr = np.array(test_metric_over_slices.obj_mean_over_time)
        #     constraints_max_arr = np.array(test_metric_over_slices.constraints_max_over_time)
        #     constraints_percentile_arr = np.array(test_metric_over_slices.constraints_percentile_over_time)
        
        #     test_metric_name = test_metric_over_slices.test_config_name
        #     n_epochs = test_metric_over_slices.epoch

        #     axs[0].plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr, '-', label = [f'{test_metric_name}'])
        #     axs[1].plot(np.arange(constraints_max_arr.shape[0]), constraints_max_arr, '-', label = [f'{test_metric_name}'])
        #     axs[2].plot(np.arange(constraints_percentile_arr.shape[0]), constraints_percentile_arr, '-', label = [f'{test_metric_name}'])
            
        #     axs[0].set_ylabel('Obj')
        #     axs[1].set_ylabel('Max constraint slack')
        #     axs[2].set_ylabel('Percentile constraint slack')

        #     for ax in axs.flat:
        #         ax.grid(True)
        #         ax.set_xlabel('Timestep (t)')
        #         if i == len(test_metrics_over_slices)-1:
        #             ax.axvline(x = self.config.T['train']-1, linestyle = '--', color = 'k', label = 'T')
        #         ax.legend()

        # fig.suptitle(f'Test evolution over timesteps (Epoch = {n_epochs})')
        # fig.tight_layout()
        # # fig.subplots_adjust(top=0.88)
        # os.makedirs(f'{save_path}', exist_ok=True)
        # plt.savefig(f'{save_path}/test_evolution_over_timesteps_epoch_{n_epochs}.png', dpi = 300)
        # plt.close()


def plot_traffic_evolution_over_slices(self, traffics, save_path = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)

    plt.rcParams.update({'font.size': fig_fontSize})

    n_networks = self.config.num_samples[self.phase]
    n_clients = self.config.n
    T_slices = self.config.T_slice[self.phase]

    data_rates_mbps = np.zeros(T_slices, n_networks, n_clients)
    slices = np.zeros_like(data_rates_mbps)

    network_str = re.compile('network_')
    client_str = re.compile('_client_')
    slice_str = re.compile('_t_slice_')
    json_str = re.compile('.json')
    for traffic in traffics:

        network_match = network_str.search(traffic)
        client_match = client_str.search(traffic)
        slice_match = slice_str.search(traffic)
        json_match = json_str.search(traffic)

        network_idx = int(traffic[network_match.end():client_match.start()])
        client_idx = int(traffic[client_match.end():slice_match.start()])
        slice_idx = int(traffic[slice_match.end():json_match.start()])

        if os.path.exists(traffic):
            with open(traffic, 'r') as file:
                traffic_dict = json.load(file)
                data_rates_mbps[slice_idx, network_idx, client_idx] = traffic_dict['data_rate_mbps']
                slices[slice_idx, network_idx, client_idx] = traffic_dict['slice.value']

    n_clients_per_plot = 5
    n_subplots = int(np.ceil(n_networks // n_clients_per_plot))
    for network_id in range(n_networks):
        plt.figure(1)
        fig_1, axs_1 = plt.subplots(n_subplots, 2, figsize = (2 * fig_colSize, n_subplots * fig_rowSize))

        for subplot in range(n_subplots):
            client_idx = range(subplot * n_clients_per_plot, min(n_clients, (subplot+1) * n_clients_per_plot))
            axs_1[subplot, 0].plot(np.arange(T_slices), data_rates_mbps[:, network_id, client_idx], label = [f'Client {idx}' for idx in client_idx])
            axs_1[subplot, 0].set_ylabel('Data rate (mbps)')
            axs_1[subplot, 1].plot(np.arange(T_slices), slices[:, network_id, client_idx], label = [f'Client {idx}' for idx in client_idx])
            axs_1[subplot, 1].set_ylabel('Slice index')
        
        for ax in axs_1.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_1.suptitle(f'Test evolution of traffics \n network #{network_id}')
        fig_1.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/traffic_evolution.png', dpi = 300)

        plt.close('all')
