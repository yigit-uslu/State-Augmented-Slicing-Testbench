import argparse
import json
import numpy as np
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

class LoadFromFile(argparse.Action):
    # def __call__ (self, parser, namespace, values, option_string = None):
    #     with values as f:
    #         # parse arguments in the file and store them in the target namespace
    #         parser.parse_args(f.read().split(), namespace)
    def __call__ (self, parser, namespace, values, option_string=None):

        try:
            with values as f:
                contents = f.read()

            # parse arguments in the file and store them in a blank namespace
            data = parser.parse_args(contents.split(), namespace=None)
            for k, v in vars(data).items():
                # set arguments in the target namespace if they have not been set yet
                if getattr(namespace, k, None) is None:
                    setattr(namespace, k, v)
        except:
            print('Config file could not be read...')
            pass


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help = 'Root path')
    parser.add_argument('--experiment_tag', type=str, default='TEST VIOLATION RATES', help = 'Append a custom description to the experiment (time) tag')
    parser.add_argument('--debug_enable', type=str2bool, default=False)
    # parser.add_argument('--test_on_train_data', type=str2bool, default=True, help = 'When toggled True, test on the same training networks.')
    # parser.add_argument('--disable_short_term_fading', type=str2bool, default=False, help = 'When toggled True, disable short term fading.')
    parser.add_argument('--load_from_chkpt_path', type=str, default = None, help = 'Load a training checkpoint.')
    parser.add_argument('--chkpt_step', type=int, default = 5, help = 'Set checkpoints to save and plot training status.')

    # Define constraint fncs
    parser.add_argument('--constraint_type', type=str, choices = ['slicewise', 'clientwise'], default = 'slicewise')
    parser.add_argument('--constraint_mean_rate', type=str2bool, default=False, help = 'Enable/disable slicewise mean-rate constraint.')
    parser.add_argument('--constraint_min_rate', type=str2bool, default=False, help = 'Enable/disable slicewise min-rate constraint.')
    parser.add_argument('--constraint_mean_latency', type=str2bool, default=False, help = 'Enable/disable slicewise mean-latency constraint.')
    parser.add_argument('--constraint_max_latency', type=str2bool, default=False, help = 'Enable/disable slicewise max-latency constraint.')
    parser.add_argument('--constraint_rate_violation_rate', type=str2bool, default=True, help = 'Enable/disable slicewise rate-violation-rate constraint.')
    parser.add_argument('--constraint_latency_violation_rate', type=str2bool, default=True, help = 'Enable/disable slicewise latency-violation-rate constraint.')

    # Weight the objective and constraint terms
    parser.add_argument('--w_obj', type=float, default=1e-2, help = 'Weight of the objective term.')
    parser.add_argument('--w_constraint_rate', type=float, default=1., help='Weight of the rate-constraints')
    parser.add_argument('--w_constraint_latency', type=float, default=1., help='Weight of the latency-constraints')
    parser.add_argument('--w_constraint_rate_violation_rate', type=float, default=1., help='Weight of the rate-violation-rate constraints')
    parser.add_argument('--w_constraint_latency_violation_rate', type=float, default=1., help='Weight of the latency-violation-rate constraints')

    # Specify constraint sensitivities
    parser.add_argument('--r_mean', type=float, default=1.1, help = 'Mean rate constraint (bps/Hz)')
    parser.add_argument('--r_min', type=float, default=0.8, help = 'Min. rate constraint (bps/Hz)') # 0.5
    parser.add_argument('--l_mean', type=float, default=0.08, help = 'Mean latency constraint (seconds)') # 0.08
    parser.add_argument('--l_max', type=float, default=0.10, help = 'Max. latency constraint (seconds)') # 0.12
    parser.add_argument('--r_min_violation_rate', type=float, default=0.1, help = 'Frequency of rate violations allowed')
    parser.add_argument('--l_max_violation_rate', type=float, default=0.05, help = 'Frequency of latency violations allowed')

    # Traffic model configurations
    parser.add_argument('--slice_reassignment_prob', type=float, default = 0.0, help = 'Probability of slice reassignment')
    parser.add_argument('--data_rate_mbps', type=str, default='5.0, 15.0; 2.0, 10.0; 5.0, 15.0; 2.0, 10.0', help = 'Data rate mbps range for different slices')
    parser.add_argument('--data_rate_random_walk_sigma', type=float, default=1.)

    # general/state-augmentation configurations
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_seed', type=int, default = 2468642, help = 'Random seed')
    # parser.add_argument('--m', type=int, default=1, help = 'Number of transmitters/APs')
    parser.add_argument('--n', type=int, default=20, help = 'Number of receivers/flows')

    parser.add_argument('--n_ht', type=str, default='6, 12', help = 'Range of clients in HT slice')
    parser.add_argument('--n_ll', type=str, default='4, 8', help = 'Range of clients in LL slice')
    parser.add_argument('--n_be', type=str, default='2, 8', help = 'Range of clients in BE slice')

    parser.add_argument('--T_slices', type=str, default='100, 500', help = 'Number of slicing windows')
    parser.add_argument('--T', type=str, default='1, 1', help = 'Number of time slots in each slice window')
    parser.add_argument('--T_scheduling_rounds', type=str, default='1', help = 'Number of round-robin scheduling rounds')
    parser.add_argument('--scheduling_substep_length', type=float, default=.1, help = 'substep_length used in round-robin scheduling (0.1 -> 100 ms)')

    parser.add_argument('--T_0', type=str, default='5', help = 'Sizes of the iteration window for averaging constraints for dual multiplier updates')
    parser.add_argument('--density_mode', type=str, choices=['var_density', 'fixed_density'], default = 'var_density', help = 'Density mode')
    parser.add_argument('--density_n_users_per_km_squared', type = float, default = 5., help = 'User density if density_mode is fixed.')
    parser.add_argument('--R', type = int, default = 2000, help = 'Network area side length in meters if density_mode is variable.')
    parser.add_argument('--num_samples', type=str, default='2, 1', help='Number of train/test samples')
    # parser.add_argument('--num_samples_test', type=int, default=64, help='Number of test samples')
    parser.add_argument('--BW_total', type=float, default=1., help = 'Total bandwidth (MHz)')
    parser.add_argument('--P_max_dBm', type=float, default = 10., help = 'Transmit power in dBm')
    parser.add_argument('--noise_PSD_dBm', type=float, default = -174., help = 'Noise PSD (dBm/Hz)')
    parser.add_argument('--speed', type=float, default = 1., help = 'Receiver speed (m/s) in channel model')
    parser.add_argument('--shadowing', type=float, default = 7., help = 'Shadowing standard dev. in channel model')
    parser.add_argument('--num_fading_paths', type=float, default = 100, help = 'Number of fading paths in channel model')

    parser.add_argument('--batch_size', type=int, default = 256, help = 'batch size')
    parser.add_argument('--n_train_epochs', type=int, default = 20, help = 'Number of training epochs')
    parser.add_argument('--lr_primal', type=float, default=1e-3, help = 'Learning rate for primal model parameters')
    # parser.add_argument('--lr_dual', type=str, default='0.5, 0.5', help = 'Learning rates for dual variables during train/test phases')
    parser.add_argument('--lr_dual', type=str, default='0.5, 0.5', help = 'Learning rates for dual variables during train/test phases')
    parser.add_argument('--dual_clamp_max', type=float, default=5., help = 'Clamp dual variables from above')
    parser.add_argument('--dual_slack', type=float, default=1e-2, help = 'Regularization slack')  
    parser.add_argument('--resample_rx_locs', action='store_true', help = 'Resample receiver locations before each slicing window')

    # Main model parameters
    parser.add_argument('--num_features_list', type=str, default = '64, 20', help = 'Number of model features in different layers')
    parser.add_argument('--lambda_transform', type=str2bool, default=False, help='If not None, transform dual multipliers before feeding to the primal policy. Currently only softmax is implemented.')
    parser.add_argument('--batch_norm', type=str2bool, default = False, help = 'Batch normalization flag')
    parser.add_argument('--dropout', type=float, default = 0.0, help = 'Dropout probability')
    # parser.add_argument('--dual_transform', type=str, default='none', help = 'Choose a dual normalization for dual multipliers before being processed by the GNN.')

    # parser.add_argument('--n_primal_iters', type=int, default=5, help = 'Number of primal iterations per epoch')
    parser.add_argument('--n_primal_iters', type=int, default=20, help = 'Number of primal iterations per epoch')
    parser.add_argument('--primal_dual_dithering_noise_level', type=float, default=0., help = 'Dither optimal dual multiplier estimates with noise')

    # state-augmentation training distribution and test init hyperparameters/configurations
    parser.add_argument('--dual_train_dist', type=str, default='uniform-mixture', choices=['normal', 'uniform', 'normal-mixture', 'uniform-mixture'], help='Type of dual multiplier distribution used to train the state-augmented policy.')
    parser.add_argument('--dual_train_dist_mean', type=float, default=0.5, help='Initial mean of dual_train_dist')
    parser.add_argument('--dual_train_dist_sigma', type=float, default=0.5, help='Std. dev. of the normal distribution, half of the width of support for the uniform distribution')
    parser.add_argument('--dual_train_dist_sigma_decay_period', type=int, default=50, help='Decay sigma period')
    parser.add_argument('--dual_train_dist_sigma_decay_factor', type=float, default=1.0, help='Decay sigma by this factor.')
    parser.add_argument('--dual_train_dist_sigma_clamp_min', type=float, default=0.1, help='Decay sigma no lower than this value.')
    parser.add_argument('--dual_train_dist_n_mixtures', type = int, default = 1, help = 'Number of mixtures in dual_train_dist')
    parser.add_argument('--dual_reg', type=float, default=0.)
    parser.add_argument('--lr_dual_train_dist', type=float, default=1.0, help='Learning rate for the optimal dual multipliers')
    parser.add_argument('--lr_dual_train_dist_decay_period', type=int, default = 5, help = 'Decay lr_dual_train_dist every $lr_dual_train_dist_decay_period$ epochs')
    parser.add_argument('--lr_dual_train_dist_decay_factor', type=float, default = 0.7, help = 'Decay lr_dual_train_dist by this factor')
    parser.add_argument('--lr_dual_train_dist_decay_enable_after_n_iters', type=int, default=0, help='Enable lr_dual_train_dist decay after this many epochs.')
    parser.add_argument('--dual_train_dist_constraint_relax_tol', type=float, default = .0, help = 'Relax the constraint for higher dual variables.')
    parser.add_argument('--dual_train_dist_update_period', type=int, default=10, help='Update period of the mean of dual_train_dist.')
    parser.add_argument('--dual_train_dist_update_n_iters', type=int, default=2, help='Number of dual optimization subiters.')
    parser.add_argument('--dual_train_dist_update_warmup_iters', type=int, default=0, help='Number of warmup epochs to wait before dual_train_dist_update.')
    parser.add_argument('--dual_train_dist_update_strategy', type=str, default='dual-dynamics', choices=['dual-dynamics', 'dual-backprop'], help = 'Either backpropogate through the constraint slacks to update the dual multipliers or run the dual dynamics to update.')
    parser.add_argument('--dual_test_init_strategy', type=str, default = 'zeros', choices=['zeros', 'adaptive'], help = 'Initialize test dual multipliers as either all zero, or by regressing against means of learned dual multiplier training distributions or by regressing against instantaneous optimal dual multipliers.')
    
    # optimizer hyperparameters
    parser.add_argument('--pgrad_clipping_constant', type=float, default = None, help = 'Clip the norms of primal gradients')
    parser.add_argument('--dgrad_clipping_constant', type=float, default = -1, help = 'Clip the max/min value of the dual gradients. If set to -1, dgrad_clipping_constant = r_min')
    parser.add_argument('--dgrad_clipping_constant_disable_after_n_iters', type=float, default = float('inf'), help = 'Disable dgrad clipping after this many iterations.')
    parser.add_argument('--dstep_clipping_constant', type=float, default = None, help = 'Clip the max/min value of dual steps (dstep = lr * dgrad). If set to -1, dstep_clipping_constant = lr_dual_train_dist * r_min. BEWARE: dgrad_clipping_constant is not disabled automatically.')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--momentum', type=float, default=.8, help='Momentum parameter of the optimizer')
    parser.add_argument('--use_primal_optimizer', type=str2bool, default=True, help = 'Use a primal optimizer')
    parser.add_argument('--use_primal_lr_scheduler', type=str2bool, default=True, help = 'Use learning rate scheduler for primal optimizer')
    parser.add_argument('--lr_primal_decay_period', type=int, default=10, help='Decay primal learning rate every $lr_primal_decay_period$ epochs')
    parser.add_argument('--lr_primal_decay_rate', type=float, default=0.7, help='Decay lr_primal by this factor')

    # Load an experiment from file
    # parser.add_argument('--file', type=open, action=LoadFromFile, help = 'Load a configuration from a file.')

    args = parser.parse_args()

    # # of transmitters = 1
    args.m = 1

    args.n_ht = args.n_ht.split(',')
    args.n_ht = [int(x) for x in args.n_ht]
    args.n_ll = args.n_ll.split(',')
    args.n_ll = [int(x) for x in args.n_ll]
    args.n_be = args.n_be.split(',')
    args.n_be = [int(x) for x in args.n_be]

    data_rate_mbps_arr = []
    data_rate_mbps_list = args.data_rate_mbps.split(';')
    for data_rate_mbps in data_rate_mbps_list:
        data_rate_mbps = data_rate_mbps.split(',')
        data_rate_mbps_arr.append([float(x) for x in data_rate_mbps])
    args.data_rate_mbps = data_rate_mbps_arr
        
    
    # Split T, T_0, num_samples, lr_dual to train and test if needed
    T_list = args.T.split(',')
    assert len(T_list) in [1,2]
    args.T = {'train': int(T_list[0]), 'test': int(T_list[-1])}

    T_slices_list = args.T_slices.split(',')
    assert len(T_slices_list) in [1,2]
    args.T_slices = {'train': int(T_slices_list[0]), 'test': int(T_slices_list[-1])}

    T_scheduling_rounds_list = args.T_scheduling_rounds.split(',')
    assert len(T_scheduling_rounds_list) in [1,2]
    args.T_scheduling_rounds = {'train': int(T_scheduling_rounds_list[0]), 'test': int(T_scheduling_rounds_list[-1])}

    T_0_list = args.T_0.split(',')
    assert len(T_0_list) in [1,2]
    args.T_0 = {'train': int(T_0_list[0]), 'test': int(T_0_list[-1])}
     
    num_samples_list = args.num_samples.split(',')
    assert len(num_samples_list) in [1,2]
    args.num_samples = {'train': int(num_samples_list[0]), 'test': int(num_samples_list[-1])}
    args.batch_size = args.batch_size if args.batch_size and args.batch_size <= min(args.num_samples['train'], args.num_samples['test']) else math.gcd(args.num_samples['train'], args.num_samples['test']) # batch size

    lr_dual_list = args.lr_dual.split(',')
    assert len(lr_dual_list) in [1,2]
    args.lr_dual = {'train': float(lr_dual_list[0]), 'test': float(lr_dual_list[-1])}

    # Pass features list as list
    args.num_features_list = [int(x) for x in args.num_features_list.split(',')]

    # Compute noise variance, P_max
    N = args.noise_PSD_dBm - 30 + 10 * np.log10(1e6 * args.BW_total)
    args.noise_var = np.power(10, N / 10) # total noise variance
    args.P_max = np.power(10, (args.P_max_dBm - 30) / 10) # maximum transmit power = 10 dBm

    # set network area side length based on the density mode
    if args.density_mode == 'var_density':
        args.R = 2000 if args.R is None else args.R
    elif args.density_mode == 'fixed_density':
        if args.density_n_users_per_km_squared is None:
            args.R = 2000 * np.sqrt(args.m / 20)
        else:
            args.R = 1e3 * np.sqrt(args.m / args.density_n_users_per_km_squared) # in meters
    else:
        raise Exception

    return args


# def make_parser():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--parse_args_enable', action='store_true', help = 'Parse the values in this function instead of running with the values in the main .py file')
#     parser.add_argument('--DEBUG_FLAG', type=bool, default=False)
#     parser.add_argument('--TRAIN_AND_TEST_ON_THE_SAME_NETWORK_FLAG', type=bool, default=False)
#     parser.add_argument('--ENABLE_LATENCY_CONSTRAINT_FLAG', type=bool, default=True)
#     parser.add_argument('--LOG_LATENCY_CONSTRAINT', type=bool, default = True, help = 'When true, use the logarithms of latencies to evaluate the latency slacks')
#     parser.add_argument('--ENABLE_RATE_CONSTRAINT_FLAG', type=bool, default=True)

#     # general/state-augmentation configurations
#     parser.add_argument('--random_seed', type=int, default = 1357531, help = 'Random seed')
#     parser.add_argument('--m', type=int, default=1, help = 'Number of transmitters/APs')
#     parser.add_argument('--n', type=int, default=10, help = 'Number of receivers/flows')
#     parser.add_argument('--T', type=int, default=100, help = 'Number of time slots within each slice')
#     parser.add_argument('--n_slices', type = int, default = 10, help = 'Number of slicing windows')
#     parser.add_argument('--density_mode', type=str, choices=['var_density', 'fixed_density'], default = 'var_density', help = 'Density mode')
#     parser.add_argument('--num_samples_train', type=int, default=2, help='Number of train samples')
#     parser.add_argument('--num_samples_test', type=int, default=2, help='Number of test samples')
#     parser.add_argument('--BW_total', type=float, default=1., help = 'Total bandwidth (MHz)')
#     parser.add_argument('--P_max_dBm', type=float, default = 10., help = 'Transmit power in dBm')
#     parser.add_argument('--noise_PSD_dBm', type=float, default = -234., help = 'Noise PSD (dBm/Hz)')
#     parser.add_argument('--batch_size', type=int, default = 64, help = 'batch size')
#     parser.add_argument('--num_epochs', type=int, default = 100, help = 'Number of training epochs')
#     parser.add_argument('--num_features_list', type=list, default = [64, 64], help = 'Number of model features in different layers')
#     parser.add_argument('--r_min', type=float, default=4., help = 'Min. rate constraint for HT flows')
#     parser.add_argument('--l_max', type=float, default=0.1, help = 'Max. latency constraint for LL flows')
#     parser.add_argument('--lr_main', type=float, default=1e-2, help = 'Learning rate for primal model parameters')
#     parser.add_argument('--lr_dual_ht', type=float, default=1.0, help = 'Learning rate for dual variables of HT flows')
#     parser.add_argument('--lr_dual_ll', type=float, default=0.1, help = 'Learning rate for dual variables of LL flows')
#     parser.add_argument('--T_0', type=int, default=5, help = 'size of the iteration window for averaging recent rates/latencies for dual updates')
#     parser.add_argument('--mu_clamp_max', type=float, default=100., help = 'Clamp dual variables') 
#     parser.add_argument('--mu_train_dist_scalar_ht', type=float, default=1., help = 'Scale mu train dist U(0,mu_train_dist_scalar) for HT flows')
#     parser.add_argument('--mu_train_dist_scalar_ll', type=float, default=1., help = 'Scale mu train dist U(0,mu_train_dist_scalar) for LL flows')
#     parser.add_argument('--resample_rx_locs', action='store_true', help = 'Resample receiver locations before each slicing window')
    
#     # scheduling / traffic model configurations
#     parser.add_argument('--T_0_scheduling', type=float, default=10, help = 'T_0 used in round-robin scheduling')
#     parser.add_argument('--substep_length', type=float, default=.1, help = 'substep_length used in round-robin scheduling (0.1 -> 100 ms)')
#     parser.add_argument('--n_ht', type=int, default=5, help = 'Number of high-throughput (HT) flows')
#     parser.add_argument('--n_ll', type=int, default=3, help = 'Number of low-latency (LL) flows')  
#     # n_be = n - n_ht - n_ll
#     parser.add_argument('--p_base', type=float, default = 0., help = 'fraction of BW that is equally distributed no matter what, must be in [0,1] range')



    
    
    
#     return parser