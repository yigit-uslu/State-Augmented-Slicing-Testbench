from collections import defaultdict, namedtuple
import torch
import numpy as np
import os
import random
import copy
from core.Network import Slice

SA_traffic_config = namedtuple('SA_traffic_config', ['data_rate_mbps', 'data_rate_random_walk_low', 'data_rate_random_walk_high', 'data_rate_random_walk_sigma', 'slice', 'slice_transition_mtrx', 'traffic_data_save_load_path'])
SA_network_config = namedtuple('SA_network_config', ['network_idx', 'clients_config', 'channel_config'])
SA_channel_config = namedtuple('SA_channel_config', ['T', 'R', 'P_max', 'noise_var', 'BW_total', 'T_scheduling_rounds', 'scheduling_substep_length', 'channel_data_save_load_path'])
SA_client_config = namedtuple('SA_client_config', ['client_idx', 'slice', 'traffic_config'])

def seed_everything(random_seed):
    # set the random seed
    os.environ['PYTHONHASHSEED']=str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def make_experiment_name(config):
    # create a string indicating the main experiment (hyper)parameters
    experiment_name = 'n_{}_T_slices_{}_num_samples_{}'.format(config.n,
                                                        config.T_slices,
                                                        config.num_samples
                                                        )
    return experiment_name


def compute_slice_transition_mtrx(base_weights, slice_reassignment_prob):
    slice_transition_mtrx = [[] for _ in range(len(Slice))]
    for i, row in enumerate(slice_transition_mtrx):
        weights = []
        for j in range(len(Slice)):
            if i == j:
                weights.append(1 - (1 - base_weights[j]) * slice_reassignment_prob)
            else:
                weights.append(base_weights[j] * slice_reassignment_prob)
        assert(abs(sum(weights) - 1.) < 1e-4)
        row.extend(weights)

    return slice_transition_mtrx


def sample_slice_weights(args):
    temp_weights = [random.choice(range(args.n_ht[0], args.n_ht[-1])), random.choice(range(args.n_ll[0], args.n_ll[-1])), random.choice(range(args.n_be[0], args.n_be[-1]))]

    while sum(temp_weights) > args.n:
        max_idx = np.where(np.argsort(temp_weights) == (len(temp_weights) - 1))[0].tolist()[0]
        temp_weights[max_idx] -= 1

    temp_weights.append(args.n - sum(temp_weights))
    base_weights = [x / sum(temp_weights) for x in temp_weights]
    slice_reassignment_prob = args.slice_reassignment_prob

    return base_weights, slice_reassignment_prob


def sample_data_rate_mbps(args, slice):
    data_rate_mbps = random.uniform(args.data_rate_mbps[slice.value][0], args.data_rate_mbps[slice.value][-1])
    return data_rate_mbps


def make_test_configs(args):
    test_configs = []

    test_config = copy.deepcopy(args)
    test_config.name = 'state-augmented-slicing'
    test_config.slicing_strategy = 'state-augmented'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    test_configs.append(test_config)

    test_config = copy.deepcopy(args)
    test_config.name = 'proportional-slicing'
    test_config.slicing_strategy = 'proportional'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    test_configs.append(test_config)

    test_config = copy.deepcopy(args)
    test_config.name = 'uniform-slicing'
    test_config.slicing_strategy = 'uniform'
    # test_config.test_on_train_data = True
    # test_config.dual_test_init_strategy = 'zeros'
    test_configs.append(test_config)

    return test_configs




def create_network_configs(args):
    '''
    This function creates the network/client configs to be used in simulating the channel and traffic demand evolutions.
    '''
    # slice_reassignment_prob = args.slice_reassignment_prob # 0.5
    # slice_transition_mtrx = [[] for _ in range(len(Slice))]
    # for i, row in enumerate(slice_transition_mtrx):
    #     temp_weights = [args.n_ht / args.n, args.n_ll / args.n, args.n_be / args.n, 1. - (args.n_ht + args.n_ll + args.n_be) / args.n]
    #     weights = []
    #     for j in range(len(Slice)):
    #         if i == j:
    #             weights.append(1 - (1 - temp_weights[j]) * slice_reassignment_prob)
    #         else:
    #             weights.append(temp_weights[j] * slice_reassignment_prob)
    #     assert(sum(weights) == 1.)
    #     row.extend(weights)

    # print('Slice transition mtrx: ', slice_transition_mtrx)


    network_configs = defaultdict(list)
    for phase in ['train', 'test']:
        for network_idx in range(args.num_samples[phase]):
            # Create network config
            clients_config = []

            base_weights, slice_reassignment_prob = sample_slice_weights(args)
            slice_transition_mtrx = compute_slice_transition_mtrx(base_weights=base_weights, slice_reassignment_prob = slice_reassignment_prob)
            slices = Slice.sample(weights = base_weights, n_samples=args.n)

            for client_idx in range(args.n):
                client_idx = client_idx
                slice = slices[client_idx]
                # slice = random.choice([Slice.HT] * args.n_ht + [Slice.LL] * args.n_ll + [Slice.BE] * args.n_be)
                data_rate_mbps = sample_data_rate_mbps(args, slice) # TO DO
                data_rate_random_walk_sigma = args.data_rate_random_walk_sigma
                traffic_config = SA_traffic_config(data_rate_mbps=data_rate_mbps,
                                                    data_rate_random_walk_low=1.,
                                                    data_rate_random_walk_high=100.,
                                                    data_rate_random_walk_sigma=data_rate_random_walk_sigma,
                                                    slice=slice,
                                                    slice_transition_mtrx=slice_transition_mtrx,
                                                    traffic_data_save_load_path=args.traffic_data_save_load_path
                                                    )
                
                client_config = SA_client_config(client_idx = client_idx, slice = slice, traffic_config = traffic_config)
                clients_config.append(client_config)

            channel_config = SA_channel_config(T = args.T,
                                               R = args.R,
                                               P_max = args.P_max,
                                               noise_var = args.noise_var,
                                               BW_total=args.BW_total,
                                               T_scheduling_rounds=args.T_scheduling_rounds,
                                               scheduling_substep_length=args.scheduling_substep_length,
                                               channel_data_save_load_path=args.channel_data_save_load_path)
            network_config = SA_network_config(network_idx = network_idx, clients_config = clients_config, channel_config = channel_config)
            network_configs[phase].append(network_config)

    return network_configs
    

def make_eval_fnc(eval_type, eval_slices, args):
    '''
    Define the obj/constraint evaluation functions and their to_metric() method which converts normalized obj/constraint
    terms to the actual metrics.
    '''
    log_tol = np.exp(-5).item()
    if eval_type == 'obj-mean-rate':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                mean_rate = torch.mean(metrics.rate[active_slice_idx])
            else:
                mean_rate = torch.zeros_like(metrics.rate[0]) # args.r_min
            return args.w_obj * (-mean_rate / args.r_mean)
        
        def metric_fnc(slack):
            return args.r_mean * -1 * (slack / args.w_obj)
        metric_fnc.__name__ = '(Mbps)'
        
        
    elif eval_type == 'obj-min-rate':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                min_rate = torch.amin(metrics.rate[active_slice_idx], dim = 0)
            else:
                min_rate = torch.zeros_like(metrics.rate[0]) # args.r_min
            return args.w_obj * (-min_rate / args.r_min)
        
        def metric_fnc(slack):
            return args.r_min * -1 * (slack / args.w_obj) # rate in Mbps
        metric_fnc.__name__ = '(Mbps)'
        
        
    elif eval_type == 'obj-mean-latency':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                mean_latency = torch.mean(metrics.latency_quantile[active_slice_idx])
            else:
                mean_latency = args.l_mean * torch.ones_like(metrics.latency_quantile[0])
            return args.w_obj * torch.log(log_tol + mean_latency / args.l_mean)
        
        def metric_fnc(slack):
            return 1000 * args.l_mean * (np.exp(slack / args.w_obj) - log_tol) # latency in ms
        metric_fnc.__name__ = '(ms)'
        
        
    elif eval_type == 'obj-max-latency':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                max_latency = torch.amax(metrics.latency_quantile[active_slice_idx], dim = 0)
            else:
                max_latency = args.l_max * torch.ones_like(metrics.latency_quantile[0])
            return args.w_obj * torch.log(log_tol + max_latency / args.l_max)
        
        def metric_fnc(slack):
            return 1000 * args.l_max * (np.exp(slack/ args.w_obj) - log_tol) # latency in ms
        metric_fnc.__name__ = '(ms)'
        

    elif eval_type == 'constraint-mean-rate':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                mean_rate = torch.mean(metrics.rate[active_slice_idx])
            else:
                mean_rate = args.r_mean * torch.ones_like(metrics.rate[0])
            return args.w_constraint_rate * (1 - mean_rate/args.r_mean)
        
        def metric_fnc(slack):
            return args.r_mean * (1 - slack / args.w_constraint_rate) # rate in Mbps
        metric_fnc.__name__ = '(Mbps)'
        
        
    elif eval_type == 'constraint-min-rate':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                min_rate = torch.amin(metrics.rate[active_slice_idx], dim = 0)
            else:
                min_rate = args.r_min * torch.ones_like(metrics.rate[0])
            return args.w_constraint_rate * (1 - min_rate/args.r_min)
        
        def metric_fnc(slack):
            return 1000 * args.r_min * (1 - slack / args.w_constraint_rate) # latency in ms
        metric_fnc.__name__ = '(ms)'
        

    elif eval_type == 'constraint-mean-latency':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                mean_latency = torch.mean(metrics.latency_quantile[active_slice_idx])
            else:
                mean_latency = args.l_mean * torch.ones_like(metrics.latency_quantile[0])
            return args.w_constraint_latency * torch.log(log_tol + mean_latency / args.l_mean)
        
        def metric_fnc(slack):
            return 1000 * args.l_mean * (np.exp(slack / args.w_constraint_latency) - log_tol) # latency in ms
        metric_fnc.__name__ = '(ms)'
        
        
    elif eval_type == 'constraint-max-latency':
        def eval_fnc(metrics, slices):
            active_slice_idx = [i for i,slice in enumerate(slices) if slice in eval_slices]
            if active_slice_idx:
                max_latency = torch.amax(metrics.latency_quantile[active_slice_idx], dim = 0)
            else:
                max_latency = args.l_max * torch.ones_like(metrics.latency_quantile[0])
            return args.w_constraint_latency * torch.log(log_tol + max_latency / args.l_max)
        
        def metric_fnc(slack):
            return 1000 * args.l_max * (np.exp(slack / args.w_constraint_latency) - log_tol) # latency in ms
        metric_fnc.__name__ = '(ms)'
        
    else:
        def eval_fnc(metrics, slices):
            return None
        
        def metric_fnc(metric):
            return None
        
    eval_fnc.__name__ = eval_type
    eval_fnc.to_metric = metric_fnc

    return eval_fnc


def make_feature_extractor(slice_features):
    all_variables = defaultdict(list)
    all_variables['all_edge_weight_l'] = [] if 'edge-weight' in slice_features else None
    all_variables['all_slice_weights'] = [] if 'slice-weight' in slice_features else None
    all_variables['all_slice_avg_data_rate_mbps'] = [] if 'slice-avg-data-rate' in slice_features else None
    def get_slice_features(channel_data, traffic_data):

        for cdata, tdata in zip(channel_data, traffic_data):

            if all_variables['all_edge_weight_l'] is not None:
                phase = next(iter(cdata))
                data = next(iter(cdata[phase]))
                data = data[0]
                y, edge_index_l, edge_weight_l, edge_index, \
                edge_weight, a, a_l, transmitters_index, num_graphs = \
                    data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                    data.weighted_adjacency, data.weighted_adjacency_l, \
                    data.transmitters_index, data.num_graphs
                
                all_variables['all_edge_weight_l'].append(edge_weight_l)


            slice_weights = torch.zeros((len(Slice)))
            slice_avg_data_rate_mbps = torch.zeros_like(slice_weights)
            for slice in Slice:
                slice_idx = [i for i in range(len(tdata)) if tdata[i].slice == slice]
                data_rates_mbps = [t.data_rate_mbps for i, t in enumerate(tdata) if i in slice_idx]

                slice_weights[slice.value] = len(slice_idx) / len(tdata)
                slice_avg_data_rate_mbps[slice.value] = sum(data_rates_mbps) / len(data_rates_mbps) if data_rates_mbps else 0.

            all_variables['all_slice_weights'].append(slice_weights) if all_variables['all_slice_weights'] is not None else None
            all_variables['all_slice_avg_data_rate_mbps'].append(slice_avg_data_rate_mbps) if all_variables['all_slice_avg_data_rate_mbps'] is not None else None

        features = torch.Tensor()
        for key in all_variables:
            if all_variables[key] is not None:
                all_variables[key] = torch.stack(all_variables[key])
                features = torch.cat((features, all_variables[key]), dim = -1)
        # all_slice_weights = torch.stack(all_slice_weights, dim = 0) if all_slice_weights is not None else None
        # all_slice_avg_data_rate_mbps = torch.stack(all_slice_avg_data_rate_mbps, dim = 0) if all_slice_avg_data_rate_mbps is not None else None

        # features = torch.cat((all_slice_weights, all_slice_avg_data_rate_mbps), dim = -1) # [batch_size, len(Slice) * 2]

        all_variables['all_edge_weight_l'] = [] if 'edge-weight' in slice_features else None
        all_variables['all_slice_weights'] = [] if 'slice-weight' in slice_features else None
        all_variables['all_slice_avg_data_rate_mbps'] = [] if 'slice-avg-data-rate' in slice_features else None

        return features
    
    return get_slice_features, len(Slice) * len(slice_features)



def make_dual_multiplier_sampler(constraint_type, args):
    if constraint_type == 'constraint-mean-rate':
        def dual_multiplier_sampler(size):
            lambdas = 5 * torch.rand(size=size, dtype=torch.float32)

            return lambdas
        
    elif constraint_type == 'constraint-min-rate':
        def dual_multiplier_sampler(size):
            lambdas = 1 * torch.rand(size=size, dtype=torch.float32)

            return lambdas
        
    elif constraint_type == 'constraint-mean-latency':
        def dual_multiplier_sampler(size):
            lambdas = 1 * torch.rand(size = size, dtype= torch.float32)

            return lambdas
        
    elif constraint_type == 'constraint-max-latency':
        def dual_multiplier_sampler(size):
            lambdas = 1 * torch.rand(size = size, dtype= torch.float32)

            return lambdas 
        
    return dual_multiplier_sampler
