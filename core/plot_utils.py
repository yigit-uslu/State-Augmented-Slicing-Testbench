from collections import defaultdict
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


def plot_train_primal_evolution_over_epochs(train_state_over_epochs, save_path = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)

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
    fig.tight_layout()
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/train_state_evolution.png', dpi = 300)
    plt.close()



def plot_test_evolution_over_epochs(self, test_states_over_epochs, test_config_names, save_path = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)
    q = kwargs.get('metric_quantile', 0.95)
    plot_actual_metrics = kwargs.get('plot_actual_metrics', True)

    plt.rcParams.update({'font.size': fig_fontSize})

    n_obj = 1
    n_samples, n_constraints = test_states_over_epochs[0][0].metrics.constraints.shape
    
    plt.figure(1)
    fig_1, axs_1 = plt.subplots(n_obj, 3, figsize = (3 * fig_colSize, n_obj * fig_rowSize))

    plt.figure(2)
    fig_2, axs_2 = plt.subplots(n_constraints, 3, figsize = (3 * fig_colSize, n_constraints * fig_rowSize))

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
        axs_1[0].plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr, '-', label = [f'{test_config_name}'])
        axs_1[1].plot(np.arange(obj_max_arr.shape[0]), obj_max_arr, '-', label = [f'{test_config_name}'])
        axs_1[2].plot(np.arange(obj_percentile_arr.shape[0]), obj_percentile_arr, '-', label = [f'{test_config_name}'])
        

        plt.figure(fig_2)
        for i in range(n_constraints):
            axs_2[i, 0].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, i], '-', label = [f'{test_config_name}'])
            axs_2[i, 1].plot(np.arange(constraints_max_arr.shape[0]), constraints_max_arr[:, i], '-', label = [f'{test_config_name}'])
            axs_2[i, 2].plot(np.arange(constraints_percentile_arr.shape[0]), constraints_percentile_arr[:, i], '-', label = [f'{test_config_name}'])


    plt.figure(fig_1)
    axs_1[0].set_ylabel('Obj (mean)')
    axs_1[1].set_ylabel('Obj (max)')
    axs_1[2].set_ylabel(f'Obj ({q} - percentile)')

    for ax in axs_1.flat:
        ax.grid(True)
        ax.set_xlabel('Iteration (k)')
        ax.legend()

    fig_1.suptitle(f'Test evolution obj mean/max/percentile \n across {n_samples} slicing networks')
    fig_1.tight_layout()
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/test_evolution_obj.png', dpi = 300)
    # plt.close(1)

    plt.figure(fig_2)
    for i in range(n_constraints):
        if plot_actual_metrics:
            axs_2[i, 0].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (mean)')
            axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (max)')
            axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} ({q} - quantile)')

        else:
            axs_2[i, 0].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (mean)')
            axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (max)')
            axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} slack ({q} - quantile)')

    for ax in axs_2.flat:
        ax.grid(True)
        ax.set_xlabel('Iteration (k)')
        ax.legend()

    if plot_actual_metrics:
        fig_2.suptitle(f'Test evolution constraint metrics mean/max/percentile \n across {n_samples} slicing networks')
    else:
        fig_2.suptitle(f'Test evolution constraint slacks mean/max/percentile \n across {n_samples} slicing networks')
    fig_2.tight_layout()
    # fig.subplots_adjust(top=0.88)
    os.makedirs(f'{save_path}', exist_ok=True)
    if plot_actual_metrics:
        plt.savefig(f'{save_path}/test_evolution_constraint_metrics.png', dpi = 300)
    else:
        plt.savefig(f'{save_path}/test_evolution_constraint_slacks.png', dpi = 300)
    plt.close()
    # plt.close(2)



def plot_test_evolution_over_slices(self, test_metrics_over_slices, test_config_names, save_path = None, network_idx = None, kwargs = None):
    fig_rowSize = kwargs.get('fig_rowSize', FIG_ROWSIZE)
    fig_colSize = kwargs.get('fig_colSize', FIG_COLSIZE)
    fig_fontSize = kwargs.get('fig_fontSize', FIG_FONTSIZE)
    q = kwargs.get('metric_quantile', 0.95)
    plot_actual_metrics = kwargs.get('plot_actual_metrics', True)

    plt.rcParams.update({'font.size': fig_fontSize})

    n_obj = 1
    n_samples, n_constraints, T_slices = test_metrics_over_slices[0][0].metrics_over_slices.constraints_over_slices.shape
    n_slices = test_metrics_over_slices[0][0].metrics_over_slices.Ps_over_slices.shape[1]

    n_epochs = len(test_metrics_over_slices[0])

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

    for test_metric_over_slices in test_metrics_over_slices:

        obj_mean_arr = [test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        obj_max_arr = [test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)].max(-1) for t in range(T_slices)]
        obj_percentile_arr = [np.quantile(test_metric_over_slices[-1].metrics_over_slices.obj_over_slices[:, max(0, t-self.config.T_slices['train']):(t+1)], q=q, axis = -1) for t in range(T_slices)]
        constraints_mean_arr = [test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].mean(-1) for t in range(T_slices)]
        constraints_max_arr = [test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)].max(-1) for t in range(T_slices)]
        constraints_percentile_arr = [np.quantile(test_metric_over_slices[-1].metrics_over_slices.constraints_over_slices[:, :, max(0, t-self.config.T_slices['train']):(t+1)], q=q, axis = -1) for t in range(T_slices)]
        Ps_arr = test_metric_over_slices[-1].metrics_over_slices.Ps_over_slices
        lambdas_arr = test_metric_over_slices[-1].metrics_over_slices.lambdas_over_slices
        constraint_violations_arr = [test_metric_over_slices[-1].metrics_over_slices.constraint_violations_over_slices[:, :, :, max(0, t-self.config.T_slices['train']):(t+1) ].mean(-1) for t in range(T_slices)]

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

        all_obj_mean_arr.append(obj_mean_arr)
        all_obj_max_arr.append(obj_max_arr)
        all_obj_percentile_arr.append(obj_percentile_arr)
        all_constraints_mean_arr.append(constraints_mean_arr)
        all_constraints_max_arr.append(constraints_max_arr)
        all_constraints_percentile_arr.append(constraints_percentile_arr)
        all_Ps_arr.append(Ps_arr)
        all_lambdas_arr.append(lambdas_arr)
        all_constraint_violations_arr.append(constraint_violations_arr)

    
    for network_id in network_idx:
    
        plt.figure(1) # obj
        fig_1, axs_1 = plt.subplots(n_obj, 1, figsize = (3 * fig_colSize, n_obj * fig_rowSize))

        plt.figure(2) # constraint slacks
        fig_2, axs_2 = plt.subplots(n_constraints, 1, figsize = (3 * fig_colSize, n_constraints * fig_rowSize))

        plt.figure(3) # Ps
        fig_3, axs_3 = plt.subplots(1, n_slices, figsize = (n_slices * fig_colSize, 1 * fig_rowSize))

        plt.figure(4) # dual multipliers
        fig_4, axs_4 = plt.subplots(n_constraints, 1, figsize = (1 * fig_colSize, n_constraints * fig_rowSize))

        plt.figure(5) # individual constraint violations 
        fig_5, axs_5 = plt.subplots(n_constraints, n_slices, figsize = (n_slices * fig_colSize, n_constraints * fig_rowSize))


        for test_config_id, test_config_name in enumerate(test_config_names):
            obj_mean_arr = all_obj_mean_arr[test_config_id]
            obj_max_arr = all_obj_max_arr[test_config_id]
            obj_percentile_arr = all_obj_percentile_arr[test_config_id]
            constraints_mean_arr = all_constraints_mean_arr[test_config_id]
            constraints_max_arr = all_constraints_max_arr[test_config_id]
            constraints_percentile_arr = all_constraints_percentile_arr[test_config_id]
            lambdas_arr = all_lambdas_arr[test_config_id]
            Ps_arr = all_Ps_arr[test_config_id]


            if plot_actual_metrics:
                obj_mean_arr[:, network_id] = self.obj_fnc.to_metric(obj_mean_arr[:, network_id])
                obj_max_arr[:, network_id] = self.obj_fnc.to_metric(obj_max_arr[:, network_id])
                obj_percentile_arr[:, network_id] = self.obj_fnc.to_metric(obj_percentile_arr[:, network_id])

                for i, constraint_fnc in enumerate(self.constraint_fncs):
                    constraints_mean_arr[:, network_id, i] = constraint_fnc.to_metric(constraints_mean_arr[:, network_id, i])
                    constraints_max_arr[:, network_id, i] = constraint_fnc.to_metric(constraints_max_arr[:, network_id, i])
                    constraints_percentile_arr[:, network_id, i] = constraint_fnc.to_metric(constraints_percentile_arr[:, network_id, i])

            plt.figure(fig_1)
            axs_1.plot(np.arange(obj_mean_arr.shape[0]), obj_mean_arr[:, network_id], '-', label = f'{test_config_name}')
            # axs_1[1].plot(np.arange(obj_max_arr.shape[0]), obj_max_arr[:, network_id], '-', label = f'{test_config_name}')
            # axs_1[2].plot(np.arange(obj_percentile_arr.shape[0]), obj_percentile_arr[:, network_id], '-', label = f'{test_config_name}')
            

            plt.figure(fig_2)
            for i in range(n_constraints):
                axs_2[i].plot(np.arange(constraints_mean_arr.shape[0]), constraints_mean_arr[:, network_id, i], '-', label = f'{test_config_name}')
                # axs_2[i, 1].plot(np.arange(constraints_max_arr.shape[0]), constraints_max_arr[:, network_id, i], '-', label = f'{test_config_name}')
                # axs_2[i, 2].plot(np.arange(constraints_percentile_arr.shape[0]), constraints_percentile_arr[:, network_id, i], '-', label = f'{test_config_name}')

            plt.figure(fig_3)
            for i in range(n_slices):
                axs_3[i].plot(np.arange(Ps_arr.shape[0]), Ps_arr[:, network_id, i], '-', label = f'{test_config_name}')

            plt.figure(fig_4)
            for i in range(n_constraints):
                axs_4[i].plot(np.arange(lambdas_arr.shape[0]), lambdas_arr[:, network_id, i], '-', label = f'{test_config_name}')

            if test_config_name == 'state-augmented-slicing':
                plt.figure(fig_5)
                for i in range(n_constraints):
                    for j in range(n_slices):
                        slice_idx = [int(client.client_idx) for client in self.network_states['test'][network_id].slice(Slice(j))]
                        axs_5[i, j].plot(constraint_violations_arr[:, network_id, i, slice_idx], '-', label = [f'Client {idx}' for idx in slice_idx])
        
        plt.figure(fig_1)
        axs_1.set_ylabel('Obj (mean)')
        # axs_1[1].set_ylabel('Obj (max)')
        # axs_1[2].set_ylabel(f'Obj ({q} - percentile)')

        # for ax in axs_1.flat:
        axs_1.grid(True)
        axs_1.set_xlabel('Slice window (t)')
        axs_1.legend()

        fig_1.suptitle(f'Test evolution obj mean/max/percentile \n network #{network_id}')
        fig_1.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_obj.png', dpi = 300)
        # plt.close(1)

        plt.figure(fig_2)
        for i in range(n_constraints):
            if plot_actual_metrics:
                axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} in {self.constraint_fncs[i].to_metric.__name__} (mean)')
            else:
                axs_2[i].set_ylabel(f'{self.constraint_fncs[i].__name__} slack (mean)')
            # axs_2[i, 1].set_ylabel(f'{self.constraint_fncs[i].__name__} (max)')
            # axs_2[i, 2].set_ylabel(f'{self.constraint_fncs[i].__name__} ({q} - quantile)')

        for ax in axs_2.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        if plot_actual_metrics:
            fig_2.suptitle(f'Test evolution constraint metrics mean/max/percentile \n network #{network_id}')
        else:
            fig_2.suptitle(f'Test evolution constraint slacks mean/max/percentile \n network #{network_id}')
        fig_2.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        if plot_actual_metrics:
            plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_metrics.png', dpi = 300)
        else:
            plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_slacks.png', dpi = 300)


        plt.figure(fig_3)
        for i in range(n_slices):
            axs_3[i].set_ylabel(f'{Slice.list()[i]}')

        for ax in axs_3.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_3.suptitle(f'Test evolution of resource allocs. \n network #{network_id}')
        fig_3.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_Ps.png', dpi = 300)


        plt.figure(fig_4)
        for i in range(n_constraints):
            axs_4[i].set_ylabel(f'{self.constraint_fncs[i].__name__}')

        for ax in axs_4.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_4.suptitle(f'Test evolution of dual multipliers \n network #{network_id}')
        fig_4.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_lambdas.png', dpi = 300)


        plt.figure(fig_5)
        for i in range(n_constraints):
            for j in range(n_slices):
                axs_5[i, j].set_ylabel(f'{self.constraint_fncs[i].__name__}, {Slice(j)}')

        for ax in axs_5.flat:
            ax.grid(True)
            ax.set_xlabel('Slice window (t)')
            ax.legend()

        fig_5.suptitle(f'Test evolution of average constraint violation indicators \n network #{network_id}')
        fig_5.tight_layout()
        # fig.subplots_adjust(top=0.88)
        os.makedirs(f'{save_path}/network_{network_id}/', exist_ok=True)
        plt.savefig(f'{save_path}/network_{network_id}/epoch_{n_epochs}_test_evolution_constraint_violations.png', dpi = 300)

        plt.close('all')



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
