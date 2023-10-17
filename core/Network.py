from collections import namedtuple
import json
import torch
import numpy as np
import math
import random
import os
from core.Slice import Slice
from core.data_gen import create_data
from core.traffic_models import ConstantBitRateModel
from core.scheduling import simulate_round_robin_scheduling
from collections import defaultdict
    

SA_scheduling_metrics = namedtuple('scheduling_metrics', ['scheduling_round', 'latency_quantile', 'latency_avg', 'throughput', 'rate', 'slice_rate'])
SA_slicing_metrics = namedtuple('slicing_metrics', ['slicing_round', 'latency_quantile', 'latency_avg', 'throughput', 'rate', 'slice_rate'])


class Network:
    '''
    A wireless network class instance
    '''
    def __init__(self, config):
        # self.config = config
        # self.time = 0.
        self.network_idx = config.network_idx
        self.clients_config = config.clients_config
        self.channel_config = config.channel_config
        self.clients = [Client(config=client_config, client_idx=idx, network_idx=self.network_idx) for idx, client_config in enumerate(self.clients_config)]
        self.channel_data_save_load_path = self.channel_config.channel_data_save_load_path
        self.channel_data = self.create_channel_data(t_slice = 0) # self.config.data # = create_data(1, self.config.n, self.config.T, self.config.R, self.config.path, self.config.num_samples, batch_size, P_max, noise_var, resample_rx_locs)
        self.slicing_metrics = []

    def create_channel_data(self, t_slice):
        save_load_path = self.channel_data_save_load_path + f'/network_{self.network_idx}_t_slice_{t_slice}.json' # TO DO
        channel_data = create_data(1, self.len, self.channel_config.T['train'], self.channel_config.R, save_load_path, {'train': 1}, 1, self.channel_config.P_max, self.channel_config.noise_var)
        return channel_data

    def step(self, p, t_slice):
        '''
        In every step, the network evolves through one slicing window by simulating round robin scheduling for T timeslots each of which lasts 100 ms. Across timeslot, the fast-fading channel component varies.
        p: Slicing decisions.
        '''
        # Update channel realization for the network
        phase = next(iter(self.channel_data))
        
        all_variables = defaultdict(list)
        for t in range(self.channel_config.T[phase]):
            scheduling_metrics = self.simulate_round_robin_scheduling(p, t) # round-robin scheduling

            all_variables['latency_quantile'].append(torch.nan_to_num_(scheduling_metrics.latency_quantile))
            all_variables['latency_avg'].append(torch.nan_to_num_(scheduling_metrics.latency_avg))
            all_variables['throughput'].append(scheduling_metrics.throughput)
            all_variables['rate'].append(scheduling_metrics.rate)
            all_variables['slice_rate'].append(scheduling_metrics.slice_rate)
        
        all_variables['latency_quantile'] = torch.stack(all_variables['latency_quantile'], dim = -1).nanquantile(q = 0.99, dim = -1)
        all_variables['latency_avg'] = torch.stack(all_variables['latency_avg'], dim = -1).nanmean(dim = -1)
        all_variables['throughput'] = torch.stack(all_variables['throughput'], dim = -1).nanmean(dim = -1)
        all_variables['rate'] = torch.stack(all_variables['rate'], dim = -1).nanmean(dim = -1)
        all_variables['slice_rate'] = torch.stack(all_variables['slice_rate'], dim = -1).nanmean(dim = -1)

        slicing_metrics = SA_slicing_metrics(slicing_round=t_slice,
                                             latency_quantile=all_variables['latency_quantile'],
                                             latency_avg=all_variables['latency_avg'],
                                             throughput=all_variables['throughput'],
                                             rate=all_variables['rate'],
                                             slice_rate=all_variables['slice_rate']
                                             )
        self.slicing_metrics.append(slicing_metrics)

        # Update traffic demands and slice types for each client
        for client in self.clients:
            client.step(t_slice = t_slice + 1)

        self.channel_data = self.create_channel_data(t_slice = t_slice + 1)

        # return slicing_metrics


    def simulate_round_robin_scheduling(self, p, t):
        DEBUG_FLAG = False
        #########################################
        ### ME: Simulate T0 scheduling rounds ###
        #########################################
        
        tol = 1e-4 # fix soonest_pkt_time and tx_time mismatch on the order of 1e-9
        device = p.device
        
        m = 1
        n = self.len # number of users/flows/applications
        total_tx_bytes = np.zeros((n,))                 # total amount of transmitted bytes per user
        packet_latency = [[] for _ in range(n)]         # list of per-packet latencies for each user
        
        pkt_queues = [[] for _ in range(n)]   ## ME
        pkt_arrival_size = np.zeros((n,))     ## ME
        
        tx_idx = 0 # self.channel_data.transmitters_index % m
        rx_idx = torch.arange(n).to(device)

        phase = next(iter(self.channel_data))
        a = next(iter(self.channel_data[phase]))[0].weighted_adjacency[0, :, tx_idx+1:, t]
        
        su_slice_rates = torch.zeros(n, device = device)
        su_rates = torch.zeros_like(su_slice_rates)
        
        # ME: Calculate rates for single user power allocation in Mbit/s
        for jj, client in enumerate(self.clients):
            if client.slice == Slice.IA:
                bw = 0.
                su_slice_rates[jj] = 0.
                su_rates[jj] = 0.
            else:
                bw = self.channel_config.BW_total * p[client.slice.value]
                su_slice_rates[jj] = bw * torch.log2(1 + self.channel_config.P_max * a[tx_idx, jj] / (bw/self.channel_config.BW_total * self.channel_config.noise_var))  # rate in Mbps
                # su_slice_rates[jj] = bw * torch.log2(1 + self.channel_config.P_max * a[tx_idx, jj] / (bw/self.channel_config.BW_total * self.channel_config.noise_var))  # rate in Mbps
                su_rates[jj] = su_slice_rates[jj] / len(self.slice(client.slice)) # len([c.slice == client.slice for c in self.clients])
        
        # simulate substeps of length substep_length
        T_scheduling_rounds = self.channel_config.T_scheduling_rounds[phase]
        substep_length = self.channel_config.scheduling_substep_length
        for t0 in range(T_scheduling_rounds): # this T_0 is different from the other T_0
            # generate arriving packets
            for jj in range(n):
                if not self.clients[jj].slice == Slice.IA:                
                    # new_pkts, pkt_arrival_size[jj]  = traffics[ii * m + jj].generate_packets(substep_length, t*T_0*substep_length + t0*substep_length)
                    new_pkts, pkt_arrival_size[jj] = self.clients[jj].traffic_model.generate_packets((t*T_scheduling_rounds + t0)*substep_length, substep_length)
                    pkt_queues[jj] += new_pkts  # might want to keep track of total num of packets and use that as a state
            if DEBUG_FLAG and any([len(pkt_queues[jj]) == 0 for jj in range(n)]):
                print('Warning, pkt_queues is empty, #network: {}, #pkt_queues_length = {}'.format(self.network_idx, [len(pkt_queues[jj]) for jj in range(n)]))

            for slice in Slice:
                slice_clients = self.slice(slice)
                slice_clients_idx = [client.client_idx for client in slice_clients]

                if slice == Slice.IA: # no scheduling for inactive slice members
                    continue
            # for slice_idx in list(set([client.slice for client in self.clients])):
            #     slice_clients = []
            #     for client in self.clients:
            #         if client.slice == slice_idx:
            #             slice_clients.append(client)

                # transmit packets in queues in random round robin order until subset_length is reached
                # tx_time = 0 # tx_time is relative to (t*T_0 + t0) * substep_length
                tx_time = torch.zeros(size = (1,), device = su_slice_rates.device) # Autograd Fix
                while tx_time <= substep_length and any(pkt_queues[jj] for jj in slice_clients_idx):
                    # Advance time to the soonest packet's time tag if it is later than the current time
                    soonest_pkt_time_tag = np.min([pkt_queues[jj][0][1] if len(pkt_queues[jj]) else np.Inf for jj in slice_clients_idx]) - (t * T_scheduling_rounds + t0) * substep_length
                    # tx_time = soonest_pkt_time_tag if soonest_pkt_time_tag > tx_time else tx_time
                    tx_time.data = torch.tensor([soonest_pkt_time_tag], device = su_slice_rates.device, dtype=torch.float32) if soonest_pkt_time_tag > tx_time else tx_time.data # Autograd FIX
                    for jj in random.sample(slice_clients_idx, len(slice_clients_idx)):
                    # for jj in np.random.permutation(m):
                        if tx_time <= substep_length and any(pkt_queues[jj] for jj in slice_clients_idx):
                            if len(pkt_queues[jj]) and pkt_queues[jj][0][1] <= tx_time + (t*T_scheduling_rounds + t0) * substep_length + tol:  # if there is packet in queue whose time tag is earlier than current time
                                tx_packet = pkt_queues[jj].pop(0)
                                # transmit_length = (tx_packet[0] * 8) / (su_rates[ii, jj].item() * 1e6)
                                transmit_length = (tx_packet[0] * 8) / (su_slice_rates[jj] * 1e6) # rate was in Mbps # NOTE: single packet every 10 ms for LL for instance -> 500 bytes every 10 ms adjust to data rate not the packet size
                                tx_time += transmit_length
                                total_tx_bytes[jj] += tx_packet[0]  # add to total transmitted bytes
                                packet_latency[jj].append((t*T_scheduling_rounds + t0)*substep_length + tx_time - tx_packet[1])  # calculate packet latency and add to list
                                
                        else:
                            break    

        if DEBUG_FLAG and any([len(packet_latency[jj]) == 0 for jj in range(n)]):
            print('Warning, empty packet_latency, #network: {}, #pkt_latency_length = {}'.format(self.network_idx, [len(packet_latency[jj]) for jj in range(n)]))

        # packet_latency_quantile = torch.tensor([torch.quantile(torch.tensor(packet_latency[jj]), 0.99) if packet_latency[jj] else float('nan') for jj in range(len(packet_latency))])
        packet_latency_quantile = torch.stack([torch.quantile(torch.stack(packet_latency[jj]).squeeze(1), 0.99) if packet_latency[jj] else torch.tensor(float('nan')).to(device) for jj in range(len(packet_latency))]) # Autograd fix
        packet_latency_avg = torch.stack([torch.mean(torch.stack(packet_latency[jj]).squeeze(1)) if packet_latency[jj] else torch.tensor(float('nan')).to(device) for jj in range(len(packet_latency))])

        
        su_throughput = 8e-6/(substep_length * T_scheduling_rounds) * torch.tensor(total_tx_bytes).to(device) # Mbits/s
        # packet_latency_all_graphs[ii, :] = packet_latency_quantile
        # su_throughput[ii, :] = 8e-6/(substep_length * T_0_scheduling) * torch.tensor(total_tx_bytes).to(device) # Mbits/s       

        scheduling_metrics = SA_scheduling_metrics(scheduling_round=t,
                                                   latency_avg=packet_latency_avg,
                                                   latency_quantile=packet_latency_quantile, # [n_clients]
                                                   throughput=su_throughput, # [n_clients]
                                                   rate=su_rates, # [n_clients]
                                                   slice_rate=su_slice_rates # [n_clients]
                                                   )
        return scheduling_metrics


    @property
    def len(self):
        return len(self.clients)

    def slice(self, Slice): # return all clients from the same slice
        return [client for client in self.clients if client.slice == Slice]
    
    # @property
    # def len(self, Slice):
    #     return len([client.slice == Slice and client.status_active == True for client in self.clients])

    # @property
    # def get(self, idx = 0):
    #     return self.clients[idx]       

class Client:
    def __init__(self, config, network_idx, client_idx):
        # self.slice = Slice(config.slice)
        self.network_idx = network_idx
        self.client_idx = client_idx
        self.traffic_config = config.traffic_config
        self.traffic_model = ConstantBitRateModel(self.traffic_config)
        self.traffic_data_save_load_path = self.traffic_config.traffic_data_save_load_path

        traffic_save_load_path = self.traffic_data_save_load_path + f'/network_{self.network_idx}_client_{self.client_idx}_t_slice_{0}.json'
        if os.path.exists(traffic_save_load_path):
            with open(traffic_save_load_path, 'r') as file:
                temp_dict = json.load(file)
                self.traffic_model.data_rate_mbps = temp_dict['data_rate_mbps']
                self.traffic_model.slice = Slice(temp_dict['slice.value'])

        else:
            with open(traffic_save_load_path, "w") as file:
                json.dump({'data_rate_mbps': self.traffic_model.data_rate_mbps,
                           'slice.value': self.traffic_model.slice.value}, file)


    @property
    def slice(self):
        return self.traffic_model.slice

    # @property
    # def status_active(self):
    #     return self.status_active
    
    # @status_active.setter
    # def status_active(self, new_status):
    #     self.status_active = new_status
    #     if new_status == False:
    #         self.data_rate_mbps = 0.
    #     else:
    #         raise NotImplementedError


    def step(self, t_slice):
        '''
        In every step, clients' traffic demands refresh and their slice type might change.
        '''
        traffic_save_load_path = self.traffic_data_save_load_path + f'/network_{self.network_idx}_client_{self.client_idx}_t_slice_{t_slice}.json'
        if os.path.exists(traffic_save_load_path):
            with open(traffic_save_load_path, 'r') as file:
                temp_dict = json.load(file) 
                self.traffic_model.data_rate_mbps = temp_dict['data_rate_mbps']
                self.traffic_model.slice = Slice(temp_dict['slice.value'])
        else:
            self.traffic_model.step(t_slice)
            with open(traffic_save_load_path, "w") as file:
                json.dump({'data_rate_mbps': self.traffic_model.data_rate_mbps,
                           'slice.value': self.traffic_model.slice.value}, file)
        

