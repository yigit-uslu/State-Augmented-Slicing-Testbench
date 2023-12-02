from collections import defaultdict, namedtuple
import random
import numpy as np
import math
import copy

from core.Slice import Slice
from core.utils import sample_data_rate_mbps
traffics = ['HT', 'LL', 'BE']

# traffic_config = namedtuple('traffic_config', ['data_rate_random_walk_low', 'data_rate_random_walk_high', 'data_rate_random_walk_sigma', 'traffic_type', 'traffic_drop_chance', 'traffic_join_chance'])
# def create_traffics(traffic_configs, args):
#     ########### Traffic model ################
# 	traffics = defaultdict(list)
# 	for phase in args.phases:
# 		traffics[phase] = [None for _ in range(args.num_batches[phase])]
# 		for batch_idx in range(args.num_batches[phase]):
# 			traffics[phase][batch_idx] = [None for t in range(args.T)]
# 			traffics_t0 = []
# 			for batch in range(args.batch_size): # num_graphs many network/traffics realizations
# 				data_rate_ht = np.random.uniform(low = traffic_configs['HT'].data_rate_random_walk_low, high = traffic_configs['HT'].data_rate_random_walk_high, size = (args.n_ht,)) # vary for each slicing epoch
# 				data_rate_be = np.random.uniform(low = traffic_configs['BE'].data_rate_random_walk_low, high = traffic_configs['BE'].data_rate_random_walk_high, size = (args.n_be,))
# 				data_rate_ll = np.random.uniform(low = traffic_configs['LL'].data_rate_random_walk_low, high = traffic_configs['LL'].data_rate_random_walk_high, size = (args.n_ll,)) # 1.2 Mbps -> 1500 bytes = 1 pkt every 10 ms

# 				# traffic_config_ll = traffic_config(data_rate_random_walk_sigma=0.01 * 0, traffic_type='LL', traffic_drop_chance=0.05 * 0, traffic_join_chance=0.1 * np.array([args.n_ht, args.n_ll, args.n_be])/args.n)
# 				# traffic_config_be = traffic_config(data_rate_random_walk_sigma=0.01 * 0, traffic_type='BE', traffic_drop_chance=0.05 * 0, traffic_join_chance=0.1 * np.array([args.n_ht, args.n_ll, args.n_be])/args.n)
# 				# traffic_config_ht = traffic_config(data_rate_random_walk_sigma=0.01 * 0, traffic_type='HT', traffic_drop_chance=0.05 * 0, traffic_join_chance=0.1 * np.array([args.n_ht, args.n_ll, args.n_be])/args.n)

# 				for i in range(args.n_ht):
# 					traffics_t0.append(ConstantBitRateModel(data_rate_ht[i], traffic_configs['HT']))
# 				for i in range(args.n_ll):
# 					traffics_t0.append(ConstantBitRateModel(data_rate_ll[i], traffic_configs['LL']))
# 				for i in range(args.n_be):
# 					traffics_t0.append(ConstantBitRateModel(data_rate_be[i], traffic_configs['BE']))
# 			traffics[phase][batch_idx][0] = traffics_t0

# 			for t in range(1, args.T):
# 				temp_traffics = copy.copy(traffics[phase][batch_idx][t-1])
# 				for idx, traffic in enumerate(temp_traffics):
# 					traffic.step(traffic_idx = idx, debug = args.DEBUG_FLAG) if t % args.T_0 else None
# 				traffics[phase][batch_idx][t] = temp_traffics
# 	return traffics
# 	########### Traffic model ################

class TrafficModel(object):
	def __init__(self, config):
		self.data_rate_mbps = config.data_rate_mbps
		self.slice = config.slice
		self.data_rate_random_walk_low = config.data_rate_random_walk_low
		self.data_rate_random_walk_high = config.data_rate_random_walk_high
		self.data_rate_random_walk_sigma = config.data_rate_random_walk_sigma
		self.slice_transition_mtrx = config.slice_transition_mtrx
		self.history = config.history

	@classmethod
	def list(cls):
		return list(map(lambda c: c.value, cls))

	def generate_packets(self, time_window):
		raise NotImplementedError
	
	def step(self):
		raise NotImplementedError


class ConstantBitRateModel(TrafficModel):
	def __init__(self, config, pkt_size_bytes=1500):
		self.traffic_data_save_load_path = config.traffic_data_save_load_path
		super(ConstantBitRateModel, self).__init__(config)

		self.pkt_size_bytes = pkt_size_bytes
		self.period = (pkt_size_bytes * 8) / (self.data_rate_mbps * 1e6) # NOTE: ME fix
		# self.period = (pkt_size_bytes * 8) / (data_rate_mbps / 1e6)

	def generate_packets(self, t, time_window):
		pkts = [(self.pkt_size_bytes,t+s) for s in np.arange(0,time_window,self.period)]     # Each pckt entry in list is tuple=(Packet size in bytes, Packet arrival time)
		total_bytes = self.pkt_size_bytes * int(time_window / self.period)                   # total amount of arriving data in bytes
		return pkts, total_bytes
	
	def step(self, t_slice = 0):
		'''
		Evolve traffic demand for each slicing window
		'''
		current_slice = self.slice
		self.slice = Slice.sample(weights = self.slice_transition_mtrx[self.slice.value], n_samples=1)[0] # updated slice type

		if current_slice == self.slice:
			self.data_rate_mbps = max(0, (self.data_rate_random_walk_sigma * np.random.randn(1) + self.data_rate_mbps).item()) # updated data rate
		else:
			self.data_rate_mbps = random.uniform(self.data_rate_random_walk_low[self.slice.value], self.data_rate_random_walk_high[self.slice.value])

	# 	if self.active:
	# 		self.data_rate_mbps += (self.data_rate_random_walk_sigma * np.random.randn(1)).item()
	# 		if np.random.uniform(low = 0, high = 1, size = (1,)) < self.traffic_drop_chance: # drop out of network
	# 			if debug:
	# 				print('Traffic #{} dropped out of the network.'.format(traffic_idx))
	# 			# self.data_rate_mbps = 0.
	# 			self.active = False
	# 	else:
	# 		if np.random.uniform(low = 0, high = 1, size = (1,)) < sum(self.traffic_join_chance): # rejoin the network
	# 			prev_traffic_type = self.traffic_type
	# 			self.traffic_type = np.random.choice(traffics, size = (1,), p = self.traffic_join_chance / sum(self.traffic_join_chance)).item()
	# 			# self.traffic_type = traffics[(self.get_traffic_type_idx() + 1) % len(traffics)] # shift the traffic type by one
	# 			# self.data_rate = 2.0
	# 			self.active = True
	# 			if debug:
	# 				print('Traffic #{} rejoined the network. Prev traffic type: {} | New traffic type: {}'.format(traffic_idx, prev_traffic_type, self.traffic_type))

	# def get_traffic_type_idx(self):
	# 	return int(np.where([self.traffic_type == traffic for traffic in traffics])[0]) if self.active else float('nan')

	# def get_data_rates_vector(self):
	# 	data_rates = [0.] * len(traffics)
	# 	if not math.isnan(self.get_traffic_type_idx()):
	# 		data_rates[self.get_traffic_type_idx()] = self.data_rate_mbps
	# 	return data_rates

	# def set_data_rate_mbps(self, data_rate_mbps):
	# 	self.data_rate_mbps = data_rate_mbps
     
		

class PoissonModel(TrafficModel):
	def __init__(self, data_rate_mbps, pkt_size_bytes=1500):
		super(PoissonModel, self).__init__(data_rate_mbps)	
		self.pkt_size_bytes = pkt_size_bytes
		self.buffer_time = 0

	def generate_packets(self, time_window):
		raise NotImplementedError