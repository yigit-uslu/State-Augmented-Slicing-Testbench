import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from torch_geometric.loader import DataLoader
from core.channel import create_channel_matrix_over_time
from utils import Data_modTxIndex, WirelessDataset, convert_channels, calc_rates, ITLinQ
import glob

DEBUG_FLAG = False

def create_data(m, n, T_eff, R, path, num_samples, batch_size, P_max, noise_var, **kwargs):
    resample_rx_locs = kwargs.get('resample_rx_locs', False)
    shuffle = kwargs.get('shuffle', True)
    relax_assoc = kwargs.get('relax_assoc', False)
    channel_data_only_path = kwargs.get('channel_data_only_path', None)
    
    if resample_rx_locs == True:
        warmup_steps = 0
    else:
        warmup_steps = 0 # 50 number of steps to stabilize the user rates
    T = T_eff + warmup_steps
    
    if os.path.exists(path):
        if DEBUG_FLAG:
            print('Data path exists. Loading from the path.')
        # baseline_rates, data_list = torch.load(path)
        data_list = torch.load(path)
        
    elif channel_data_only_path is not None:
        # glob match
        raise NotImplemented

    else:
        if DEBUG_FLAG:
            print('Data path does not exist. Creating data...')
        # create datasets
        H = defaultdict(list)
        H_l = defaultdict(list)
        A = dict() # reshaped instantaneous weighted adjacency matrix
        A_l = dict() # reshaped large-scale weighted adjacency matrix
        associations = dict()
        for phase in num_samples:
            for _ in range(num_samples[phase]):
                h, h_l = create_channel_matrix_over_time(m, n, T, R, relax_assoc = relax_assoc)  # varying H_l support
                H[phase].append(h)
                H_l[phase].append(h_l)
            H[phase] = np.stack(H[phase])
            H_l[phase] = np.stack(H_l[phase])
            if relax_assoc:
                associations[phase] = np.stack([np.eye(H_l[phase].shape[1]) for _ in range(num_samples[phase])])
            else:
                associations[phase] = (H_l[phase] == np.max(H_l[phase], axis=1, keepdims=True))

            if not resample_rx_locs:
                # reshape the channel matrices to get the weighted adjacency matrices as the basis for GNNs
                # instantaneous channel
                A[phase] = np.zeros((num_samples[phase], m+n, m+n, T))
                A[phase][:, :m, m:, :] = np.expand_dims(associations[phase], 3) * H[phase]
                A[phase][:, m:, :m, :] = np.transpose((np.expand_dims((1 - associations[phase]), 3) * H[phase]), (0, 2, 1, 3))
                # long-term channel
                A_l[phase] = np.zeros((num_samples[phase], m+n, m+n))
                A_l[phase][:, :m, m:] = associations[phase] * H_l[phase]
                A_l[phase][:, m:, :m] = np.transpose(((1 - associations[phase]) * H_l[phase]), (0, 2, 1))
            else:
                # reshape the channel matrices to get the weighted adjacency matrices as the basis for GNNs
                # instantaneous channel
                A[phase] = np.zeros((num_samples[phase], m+n, m+n, T))
                A[phase][:, :m, m:, :] = associations[phase] * H[phase]
                A[phase][:, m:, :m, :] = np.transpose(((1 - associations[phase]) * H[phase]), (0, 2, 1, 3))
                # long-term channel
                A_l[phase] = np.zeros((num_samples[phase], m+n, m+n, T))
                A_l[phase][:, :m, m:, :] = associations[phase] * H_l[phase]
                A_l[phase][:, m:, :m, :] = np.transpose(((1 - associations[phase]) * H_l[phase]), (0, 2, 1, 3))
                

        # create PyG graphs
        data_list = defaultdict(list)
        y = torch.ones(n, 1)
        snr = P_max / noise_var

        for phase in H:
            for i in range(num_samples[phase]):
                a, a_l, h, h_l = A[phase][i], A_l[phase][i], H[phase][i], H_l[phase][i]

                if not resample_rx_locs:
                    if relax_assoc:
                        serving_transmitters = torch.Tensor(np.argmax(np.eye(h_l.shape[0]), axis=0)).to(torch.long)
                    else:
                        serving_transmitters = torch.Tensor(np.argmax(h_l, axis=0)).to(torch.long)

                    weighted_adjacency = torch.Tensor(a).unsqueeze(0)
                    weighted_adjacency_l = torch.Tensor(a_l).unsqueeze(0)
                    gg = ((1 - associations[phase][i]) * h_l)[serving_transmitters] + np.eye(n) * h_l[serving_transmitters]
                    normalized_log_channel_matrix = convert_channels(gg, snr)
                    edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
                all_edge_indices = []
                all_edge_weights = []
                all_edge_indices_l = [] # resample rx locs
                all_edge_weights_l = [] # resample rx locs
                for t in range(T):

                    if t < warmup_steps:
                          p = P_max * torch.ones(m)
                          gamma = torch.zeros(n)
                          selected_rxs = []
                          for tx in range(m):
                              associated_receivers = np.where(weighted_adjacency[0, tx , m:, 0].detach().cpu().numpy() > 0)[0]
                              selected_receiver = associated_receivers[t % len(associated_receivers)]
                              selected_rxs.append(selected_receiver)
                          selected_rxs = np.array(selected_rxs)
                          gamma[selected_rxs] = 1
                          sampled_gamma = gamma
                          rates = calc_rates(p, sampled_gamma, weighted_adjacency[:, :, :, t], noise_var)

                    else:
                        if not resample_rx_locs:
                            gg = ((1 - associations[phase][i]) * h[:, :, t])[serving_transmitters] + np.eye(n) * h[:, :, t][serving_transmitters]
                            normalized_log_channel_matrix = convert_channels(gg, snr)
                            edge_index_t, edge_weights = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
                            all_edge_indices.append(edge_index_t)
                            all_edge_weights.append(edge_weights.float())
                        else:
                            serving_transmitters = torch.Tensor(np.argmax(h_l[:, :, t], axis=0)).to(torch.long)
                            gg = ((1 - associations[phase][i][:, :, t]) * h[:, :, t])[serving_transmitters] + np.eye(n) * h[:, :, t][serving_transmitters]
                            normalized_log_channel_matrix = convert_channels(gg, snr)
                            edge_index_t, edge_weights = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
                            all_edge_indices.append(edge_index_t) 
                            all_edge_weights.append(edge_weights.float())
                            
                            weighted_adjacency = torch.Tensor(a).unsqueeze(0)
                            weighted_adjacency_l = torch.Tensor(a_l).unsqueeze(0)
                            
                            gg = ((1 - associations[phase][i][:, :, t]) * h_l[:, :, t])[serving_transmitters] + np.eye(n) * h_l[:, :, t][serving_transmitters]
                            normalized_log_channel_matrix = convert_channels(gg, snr)
                            edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
                            all_edge_indices_l.append(edge_index_l)
                            all_edge_weights_l.append(edge_weight_l.float())
                               

                if not resample_rx_locs:
                    data_list[phase].append(Data_modTxIndex(y=y,
                                                            edge_index_l=edge_index_l,
                                                            edge_weight_l=edge_weight_l.float(),
                                                            edge_index=all_edge_indices,
                                                            edge_weight=all_edge_weights,
                                                            weighted_adjacency=weighted_adjacency,
                                                            weighted_adjacency_l=weighted_adjacency_l,
                                                            transmitters_index=serving_transmitters,
                                                            num_nodes=n,
                                                            m=m,
                                                            )
                                        )
                else:
                    data_list[phase].append(Data_modTxIndex(y=y,
                                                            edge_index_l=all_edge_indices_l,
                                                            edge_weight_l=all_edge_weights_l,
                                                            edge_index=all_edge_indices,
                                                            edge_weight=all_edge_weights,
                                                            weighted_adjacency=weighted_adjacency,
                                                            weighted_adjacency_l=weighted_adjacency_l,
                                                            transmitters_index=serving_transmitters,
                                                            num_nodes=n,
                                                            m=m,
                                                            )
                                        )

        # # calculate baseline rates for test phase
        # phase = 'test'
        # baseline_rates = defaultdict(list)
            
        # for alg in ['ITLinQ', 'FR']:
        #     print(alg)
        #     for i in tqdm(range(len(H[phase]))):
        #         a = A[phase][i]
        #         weighted_avg_rates = 1e-10 * np.ones(n)
        #         mean_rates = np.zeros(n)
        #         for t in range(T):
        #             current_S = P_max * np.sum(a[:m, m:, t], axis=0)
        #             current_I = P_max * np.sum(a[m:, :m, t], axis=1)
        #             current_rates = np.log2(1 + current_S / (noise_var + current_I))
        #             PFs = current_rates / weighted_avg_rates
        #             selected_rxs = []
        #             for tx in range(m):
        #                 if t < warmup_steps:
        #                     associated_receivers = np.where(associations[phase][i][tx, :] > 0)[0]
        #                     selected_receiver = associated_receivers[t % len(associated_receivers)]
        #                 else:
        #                     masked_PFs = (associations[phase][i][tx, :] > 0) * PFs
        #                     selected_receiver = np.argmax(masked_PFs)
        #                 selected_rxs.append(selected_receiver)
        #             h = H[phase][i][:, selected_rxs, t]

        #             if t < warmup_steps:
        #                 p = P_max * np.ones(m)
        #             else:
        #                 if alg == 'ITLinQ':
        #                     p = ITLinQ(h, P_max, noise_var, PFs[selected_rxs])
        #                 elif alg == 'FR':
        #                     p = P_max * np.ones(m)
        #                 else:
        #                     raise Exception

        #             h_power_adjusted = np.expand_dims(p, 1) * h
        #             S = np.diag(h_power_adjusted)
        #             I = np.sum(h_power_adjusted, axis=0) - S
        #             rates = np.zeros(n)
        #             rates[selected_rxs] = np.log2(1 + S / (noise_var + I))
        #             if t >= warmup_steps:
        #                 mean_rates += rates
        #         mean_rates /= (T - warmup_steps)
        #         baseline_rates[alg].extend(mean_rates.tolist())

        # torch.save([baseline_rates, data_list], path)
        torch.save(data_list, path)

    # dataloaders
    loader = {}
    for phase in data_list:
        if shuffle == True:
            loader[phase] = DataLoader(WirelessDataset(data_list[phase]), batch_size=batch_size, shuffle=(phase == 'train'))
        else:
            loader[phase] = DataLoader(WirelessDataset(data_list[phase]), batch_size=batch_size, shuffle=False)
        # loader[phase] = DataLoader(WirelessDataset(data_list[phase]), batch_size=batch_size, shuffle=False)
    return loader
    # return loader, baseline_rates