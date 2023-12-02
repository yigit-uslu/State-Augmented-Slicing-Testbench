from core.Network import Slice
from core.StateAugmentation import StateAugmentedSlicingAlgorithm
from core.config import make_parser
from argparse import Namespace
import os
import time
import json
import torch

from core.model import MLP
from core.utils import make_constraint_fncs_and_lambda_samplers, make_dual_multiplier_sampler, make_eval_fnc, make_experiment_name, make_feature_extractor, make_logger, make_test_configs, seed_everything, create_network_configs


def main(**kwargs):
    if 'args' in kwargs:
        print('Copying cmdline args from dictionary passed to main()...')
        args = kwargs.get('args')
        if not isinstance(args, Namespace):
            args = Namespace(**args)
    elif 'file' in kwargs:
        file = kwargs.get('file')
        with open(file) as f:
            print(f'Loading cmdline arguments from file {file}...')
            args_dict = json.load(f)
            args = Namespace(**args_dict)
    else:
        print('Parsing cmdline arguments...')
        args = make_parser()
        
    seed_everything(args.random_seed)

    # create folders to save the data, results, and final model
    os.makedirs(f'{args.root}/data', exist_ok=True)
    os.makedirs(f'{args.root}/results', exist_ok=True)
    # os.makedirs(f'{args.root}/models', exist_ok=True)

    # create a string indicating the main experiment (hyper)parameters
    experiment_name = make_experiment_name(args)
    args.save_dir = experiment_name + f'/{time.time()} -- {args.experiment_tag}'

    args.channel_data_save_load_path = f'{args.root}/results/{args.save_dir}/channel_data'
    args.traffic_data_save_load_path = f'{args.root}/results/{args.save_dir}/traffic_data'
        
    # Create more folders and save the parsed configuration
    os.makedirs(f'{args.root}/results/{args.save_dir}/plots', exist_ok=True)
    os.makedirs(f'{args.root}/results/{args.save_dir}/train_chkpts', exist_ok=True)
    os.makedirs(f'{args.channel_data_save_load_path}', exist_ok=True)
    os.makedirs(f'{args.traffic_data_save_load_path}', exist_ok=True)
    with open(f'{args.root}/results/{args.save_dir}/config.json', 'w') as f:
        json.dump(vars(args), f, indent = 6)

    # Create network configs to initialize wireless networks
    network_configs = create_network_configs(args)

    # Create feature extractor, obj and constraint eval functions
    feature_extractor, n_features = make_feature_extractor(['slice-weight', 'slice-avg-data-rate'], args)
    obj = make_eval_fnc(eval_type = 'obj-mean-rate', eval_slices = [Slice.BE], args=args)

    constraints, lambda_samplers = make_constraint_fncs_and_lambda_samplers(args)
    
    args.num_features_list = [n_features + len(constraints)] + args.num_features_list

    # set the computation device and create the model using a GNN parameterization
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Device: ', args.device)
    model = MLP(args.num_features_list, lambda_transform = args.lambda_transform, batch_norm = args.batch_norm).to(args.device)

    # Make test configs
    test_configs = make_test_configs(args)

    # Log print statements to a logs.txt file
    loggers = [make_logger(f'{args.root}/results/{args.save_dir}/logs.txt')]
    
    if args.load_from_chkpt_path is None:
        all_epoch_results = None
    else:
        train_chkpt = torch.load(args.load_from_chkpt_path)
        all_epoch_results = train_chkpt['all_epoch_results']

    sa_learner = StateAugmentedSlicingAlgorithm(model=model,
                                                config=args,
                                                network_configs=network_configs,
                                                feature_extractor=feature_extractor,
                                                loggers=loggers,
                                                obj=obj,
                                                constraints=constraints,
                                                lambda_samplers = lambda_samplers,
                                                all_epoch_results=all_epoch_results)
    sa_learner.fit(chkpt_step=args.chkpt_step,
                   save_path=f'{args.root}/results/{args.save_dir}',
                   test_configs=test_configs)

if __name__ == '__main__':
    main(file = 'data/experiment_3.json')