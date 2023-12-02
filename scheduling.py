import numpy as np
import torch
import math
import random

def simulate_round_robin_scheduling(traffics, p, graph_idx, transmitters_index, t, a, args):
    
    #########################################
    ### ME: Simulate T0 scheduling rounds ###
    #########################################
    
    tol = 1e-4 # fix soonest_pkt_time and tx_time mismatch on the order of 1e-9
    
    n = len(traffics) # number of users/flows/applications
    total_tx_bytes = np.zeros((n,))                 # total amount of transmitted bytes per user
    packet_latency = [[] for _ in range(n)]         # list of per-packet latencies for each user
    
    pkt_queues = [[] for _ in range(n)]   ## ME
    pkt_arrival_size = np.zeros((n,))     ## ME
    
    tx_idx = transmitters_index % args.m
    rx_idx = torch.arange(n).to(args.device)
    n_traffic_idx = [[], [], []] if not args.algo == 'no-slicing' else [[],]
    
    su_slice_rates = torch.zeros(n, device = args.device)
    su_rates = torch.zeros_like(su_slice_rates)
    
    # ME: Calculate rates for single user power allocation in Mbit/s
    for jj in range(n):
        if args.algo == 'no-slicing':
            traffic_type_idx = 0
        else:
            traffic_type_idx = traffics[jj].get_traffic_type_idx()
        if not math.isnan(traffic_type_idx): # only active applications
            n_traffic_idx[traffic_type_idx].append(jj)
            # calculate bandwidth (BW) per flow = BW per traffic  (should we normalize by the number of flows?)
            bw = args.BW_total * p[traffic_type_idx]
            su_slice_rates[jj] = bw * torch.log2(1 + args.P_max * a[tx_idx, rx_idx][jj] / (bw/args.BW_total * args.noise_var))  # rate in Mbps
    
    # normalize slicing rates by the number of applications in a given slice
    for traffic_idx in n_traffic_idx:
        for jj in traffic_idx:
            su_rates[jj] = su_slice_rates[jj] / len(traffic_idx)
    
    # simulate substeps of length substep_length
    for t0 in range(args.T_0_scheduling): # this T_0 is different from the other T_0
        # generate arriving packets
        for jj in range(n):
            if not math.isnan(traffics[jj].get_traffic_type_idx()):                
                # new_pkts, pkt_arrival_size[jj]  = traffics[ii * m + jj].generate_packets(substep_length, t*T_0*substep_length + t0*substep_length)
                new_pkts, pkt_arrival_size[jj] = traffics[jj].generate_packets((t*args.T_0_scheduling + t0)*args.substep_length, args.substep_length)
                pkt_queues[jj] += new_pkts  # might want to keep track of total num of packets and use that as a state
        if args.DEBUG_FLAG and any([len(pkt_queues[jj]) == 0 for jj in range(n)]):
            print('Warning, pkt_queues is empty, #network: {}, #pkt_queues_length = {}'.format(graph_idx, [len(pkt_queues[jj]) for jj in range(n)]))

        for idx in np.arange(len(n_traffic_idx)): # parallelize this across different traffic types
            # transmit packets in queues in random round robin order until subset_length is reached
            # tx_time = 0 # tx_time is relative to (t*T_0 + t0) * substep_length
            tx_time = torch.zeros(size = (1,), device = su_slice_rates.device) # Autograd Fix
            while tx_time <= args.substep_length and any(pkt_queues[jj] for jj in n_traffic_idx[idx]):
                # Advance time to the soonest packet's time tag if it is later than the current time
                soonest_pkt_time_tag = np.min([pkt_queues[jj][0][1] if len(pkt_queues[jj]) else np.Inf for jj in n_traffic_idx[idx]]) - (t * args.T_0_scheduling + t0) * args.substep_length
                # tx_time = soonest_pkt_time_tag if soonest_pkt_time_tag > tx_time else tx_time
                tx_time.data = torch.tensor([soonest_pkt_time_tag], device = su_slice_rates.device) if soonest_pkt_time_tag > tx_time else tx_time.data # Autograd FIX
                for jj in random.sample(n_traffic_idx[idx], len(n_traffic_idx[idx])):
                # for jj in np.random.permutation(m):
                    if tx_time <= args.substep_length and any(pkt_queues[jj] for jj in n_traffic_idx[idx]):
                        if len(pkt_queues[jj]) and pkt_queues[jj][0][1] <= tx_time + (t*args.T_0_scheduling + t0) * args.substep_length + tol:  # if there is packet in queue whose time tag is earlier than current time
                            tx_packet = pkt_queues[jj].pop(0)
                            # transmit_length = (tx_packet[0] * 8) / (su_rates[ii, jj].item() * 1e6)
                            transmit_length = (tx_packet[0] * 8) / (su_slice_rates[jj] * 1e6) # rate was in Mbps # NOTE: single packet every 10 ms for LL for instance -> 500 bytes every 10 ms adjust to data rate not the packet size
                            tx_time += transmit_length
                            total_tx_bytes[jj] += tx_packet[0]  # add to total transmitted bytes
                            packet_latency[jj].append((t*args.T_0_scheduling + t0)*args.substep_length + tx_time - tx_packet[1])     # calculate packet latency and add to list
                            
                    else:
                        break      
        ##########################################
        ##########################################
    if args.DEBUG_FLAG and any([len(packet_latency[jj]) == 0 for jj in range(n)]):
        print('Warning, empty packet_latency, #network: {}, #pkt_latency_length = {}'.format(graph_idx, [len(packet_latency[jj]) for jj in range(n)]))

    # packet_latency_quantile = torch.tensor([torch.quantile(torch.tensor(packet_latency[jj]), 0.99) if packet_latency[jj] else float('nan') for jj in range(len(packet_latency))])
    packet_latency_quantile = torch.stack([torch.quantile(torch.stack(packet_latency[jj]).squeeze(1), 0.99) if packet_latency[jj] else float('nan') for jj in range(len(packet_latency))]) # Autograd fix

    
    su_throughput = 8e-6/(args.substep_length * args.T_0_scheduling) * torch.tensor(total_tx_bytes).to(args.device) # Mbits/s
    # packet_latency_all_graphs[ii, :] = packet_latency_quantile
    # su_throughput[ii, :] = 8e-6/(substep_length * T_0_scheduling) * torch.tensor(total_tx_bytes).to(device) # Mbits/s
    
    return {'latency': packet_latency_quantile,
            'throughput': su_throughput,
            'rate': su_rates,
            'slice_rate': su_slice_rates
            }