import random
from tqdm import trange
from core.Network import Network
import torch
import numpy as np
from collections import defaultdict, namedtuple
import core.plot_utils
from core.Slice import Slice
import glob
from joblib import Parallel, delayed
import time

SA_primal_optim_state = namedtuple("SA_primal_optim_state", ['epoch', 'Lagrangian', 'obj', 'constraints', 'pgrad_norm'])
SA_dual_optim_state = namedtuple("SA_dual_optim_state", ['epoch', 'mu', 'mu_over_time', 'obj_over_slices', 'constraints_over_slices'])
SA_metrics_over_slices = namedtuple("SA_metrics_over_slices", ['epoch', 'obj_over_slices', 'constraints_over_slices', 'Ps_over_slices', 'lambdas_over_slices',
                                                               'constraint_violations_over_slices', 'slicing_metrics_over_slices'])

SA_train_state = namedtuple("SA_train_state",['epoch', 'primal_optim_state', 'primal_metrics_over_slices', 'dual_optim_state', 'dual_metrics_over_slices'])
SA_test_state = namedtuple("SA_test_state", ['epoch', 'metrics', 'metrics_over_slices'])


class StateAugmentedSlicingAlgorithm(object):
    def __init__(self,
                 model,
                 config,
                 network_configs,
                 loggers = None,
                 feature_extractor = None,
                 obj = None,
                 constraints = None,
                 lambda_samplers = None,
                 all_epoch_results = None):
        self.phase = 'train'
        self.model = model
        self.config = config
        self.loggers = loggers

        self.network_states = {
            'train': [Network(network_config) for network_config in network_configs['train']],
            'test': [Network(network_config) for network_config in network_configs['test']]
        }

        self.all_epoch_results = all_epoch_results if all_epoch_results is not None else defaultdict(list)

        self.feature_extractor = feature_extractor
        self.obj_fnc = obj
        self.constraint_fncs = constraints
        self.lambdas = torch.zeros(len(self.network_states['test']), len(self.constraint_fncs)) # estimates of optimal dual multipliers
        self.lambda_samplers = lambda_samplers

        if config.use_primal_optimizer:
            self.primal_optim = torch.optim.SGD(self.model.parameters(), lr = self.config.lr_primal, weight_decay=self.config.weight_decay, momentum=self.config.momentum)
            # self.primal_optim = torch.optim.Adam(self.model.parameters(),
            #                                     lr = self.config.lr_primal,
            #                                     weight_decay = self.config.weight_decay)
        else:
            self.primal_optim = None
        
        if config.use_primal_lr_scheduler and self.primal_optim is not None:
            self.primal_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.primal_optim, self.config.lr_primal_decay_period, self.config.lr_primal_decay_rate)
        else:
            self.primal_lr_scheduler = None

        if config.dgrad_clipping_constant == -1:
            self.config.dgrad_clipping_constant = config.r_min
        if config.dstep_clipping_constant == -1:
            self.config.dstep_clipping_constant = config.lr_dual_train_dist * config.r_min
        
        
    def fit(self, chkpt_step = None, save_path = None, test_configs = None, **kwargs):
        plot_actual_metrics = kwargs.get('plot_actual_metrics', True)
        save_subplots = kwargs.get('save_subplots', False)

        for epoch in trange(len(self.all_epoch_results['train_state']), self.config.n_train_epochs):
            train_state = self.step(epoch)
            if len(self.all_epoch_results['train_state']) and False:
                self.all_epoch_results['train_state'][-1] = train_state
            else:
                self.all_epoch_results['train_state'].append(train_state)

            for test_config in test_configs:
                
                if not epoch == 0 and test_config.name in ['proportional-slicing', 'uniform-slicing']:
                    self.all_epoch_results['test_state', test_config.name].append(self.all_epoch_results['test_state', test_config.name][-1])
                else:
                    test_state = self.test(epoch, test_config) # test
                    self.all_epoch_results['test_state', test_config.name].append(test_state)
                    
            # Save training checkpoint
            if (epoch == self.config.n_train_epochs-1 or (epoch + 1) % chkpt_step == 0):
                train_chkpt = {
                    'epoch': epoch,
                    'model': self.model,
                    'model_state_dict': self.model.state_dict(),
                    'all_epoch_results': self.all_epoch_results,
                    'config': self.config
                }
                model_chkpt = {
                    'model': self.model,
                    'model_state_dict': self.model.state_dict(),
                }
                all_epoch_results_chkpt = {
                    'all_epoch_results': self.all_epoch_results
                }
                train_chkpt['primal_optim_state_dict'] = self.primal_optim.state_dict() if self.primal_optim is not None else None
                torch.save(train_chkpt, f'{save_path}/train_chkpts/train_chkpt_{epoch}.pt')
                torch.save(model_chkpt, f'{save_path}/train_chkpts/model_chkpt_{epoch}.pt')
                torch.save(all_epoch_results_chkpt, f'{save_path}/train_chkpts/all_epoch_results_chkpt_{epoch}.pt')

                # if len(self.all_epoch_results['test_state', test_config.name]) and False:
                #     self.all_epoch_results['test_state', test_config.name][-1] = test_state
                # else:
                #     self.all_epoch_results['test_state', test_config.name].append(test_state)
                # # self.all_epoch_results['test_state', test_config.name].append(test_state)

            # if epoch == 0:
            #     self.plot_traffic_evolution_over_slices(traffics = glob.glob(f'{save_path}/traffic_data/*.json'),
            #                                             save_path = save_path + '/plots/traffic_evolution')

            self.plot_train_primal_evolution_over_epochs(self.all_epoch_results['train_state'], save_path = save_path + '/plots/train_primal_evolution/')
            self.plot_test_evolution_over_epochs([self.all_epoch_results['test_state', _.name] for _ in test_configs],
                                                 test_config_names=[_.name for _ in test_configs],
                                                 save_path=save_path + '/plots/test_evolution/',
                                                 plot_actual_metrics = plot_actual_metrics)
            
            if (epoch + 1) == self.config.n_train_epochs or (epoch + 1) % chkpt_step == 0:
                k_networks = self.config.num_samples['test'] if (epoch + 1) == self.config.n_train_epochs else 5
                self.plot_test_evolution_over_slices([self.all_epoch_results['test_state', _.name] for _ in test_configs],
                                                        test_config_names = [_.name for _ in test_configs],
                                                        save_path=save_path + '/plots/test_evolution_over_slices/',
                                                        network_idx = None, #random.sample(range(self.config.num_samples['test']), k = min(k_networks, self.config.num_samples['test'])),
                                                        plot_actual_metrics = plot_actual_metrics,
                                                        save_subplots = save_subplots)
                

            # if (epoch == self.config.n_train_epochs-1 or (epoch + 1) % chkpt_step == 0):
            #     train_chkpt = {
            #         'epoch': epoch,
            #         'model': self.model,
            #         'model_state_dict': self.model.state_dict(),
            #         'all_epoch_results': self.all_epoch_results,
            #         'config': self.config
            #     }
            #     train_chkpt['primal_optim_state_dict'] = self.primal_optim.state_dict() if self.primal_optim is not None else None
            #     torch.save(train_chkpt, f'{save_path}/train_chkpts/chkpt_{epoch}.pt')


    def step(self, epoch):
        primal_optim_state, primal_metrics_over_slices = self.primal_step(epoch)
        # dual_optim_state, dual_metrics_over_slices = self.dual_step(epoch = epoch, phase='test')
        dual_optim_state, dual_metrics_over_slices = (None, None)

        train_state = SA_train_state(epoch=epoch,
                                     primal_optim_state=primal_optim_state,
                                     primal_metrics_over_slices=primal_metrics_over_slices,
                                     dual_optim_state=dual_optim_state,
                                     dual_metrics_over_slices=dual_metrics_over_slices)
        return train_state


    def dual_step(self, epoch, phase):
        '''
        During training, we can execute the state-augmented policy offline for the training networks to observe the
        dual multiplier trajectories.
        '''
        
        return None, None

        self.phase = phase
        n_constraints = len(self.constraint_fncs)
        n_slices = self.model.layers[-1].out_features # number of slice types that compete for bandwidth
        T_slices = self.config.T_slices[self.phase] # constraints and obj are averaged over T_slices slices.
        n_clients = self.network_states[self.phase][0].len

        all_variables = defaultdict(list)
        all_variables['obj'] = torch.zeros((self.config.num_samples[self.phase]), dtype=torch.float32).to(self.config.device)
        all_variables['constraints'] = torch.zeros((self.config.num_samples[self.phase], n_constraints), dtype=torch.float32).to(self.config.device)

        all_variables['obj_over_slices'] = torch.zeros((self.config.num_samples[self.phase], T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['constraints_over_slices'] = torch.zeros((self.config.num_samples[self.phase], n_constraints, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['lambdas_over_slices'] = torch.zeros((self.config.num_samples[self.phase], n_constraints, T_slices), dtype=torch.float32).to(device='cpu')
        all_variables['Ps_over_slices'] = torch.zeros((self.config.num_samples[self.phase], n_slices, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['constraint_violations_over_slices'] = torch.zeros((self.config.num_samples[self.phase], n_constraints, n_clients, T_slices), dtype=torch.bool).to(self.config.device)

        lambdas_all = torch.zeros_like(all_variables['constraints']) # zero init
    
        self.model = self.model.to(self.config.device)
        self.model.eval()

        batches = torch.randperm(len(self.network_states[self.phase]))
        batches = torch.split(batches, self.config.batch_size)

        for batch_sample_idx in batches:
            self.model.zero_grad()
            batch_sample_idx = batch_sample_idx.tolist()
            lambdas = lambdas_all[batch_sample_idx].to(self.config.device)

            obj_over_slices, constraints_over_slices, Ps_over_slices, lambdas_over_slices, all_constraint_violations = self.evaluate(lambdas = lambdas, Trange = trange(T_slices), batch_sample_idx=batch_sample_idx)

            all_Ps = torch.stack(Ps_over_slices, dim = -1) # [batch_size, 3, T_slices]
            all_obj_terms = torch.stack(obj_over_slices, dim = -1) #[batch_size, T_slices]
            all_constraint_terms = torch.stack(constraints_over_slices, dim = -1) #[batch_size, n_constraints, T_slices]
            all_lambdas = torch.stack(lambdas_over_slices, dim = -1) # [batch_size, n_constraints, T_slices]
            all_constraint_violations = torch.stack(all_constraint_violations, dim = -1) # [batch_size, n_constraints, T_slices, n_clients]

            obj = torch.mean(all_obj_terms, dim = -1)
            constraint_slacks = torch.mean(all_constraint_terms, dim = -1)

            all_variables['obj'][batch_sample_idx] = obj # [batch_size]
            all_variables['constraints'][batch_sample_idx] = constraint_slacks # [batch_size, n_constraints]
            all_variables['obj_over_slices'][batch_sample_idx] = all_obj_terms # [batch_size, T_slices]
            all_variables['constraints_over_slices'][batch_sample_idx] = all_constraint_terms # [batch_size, n_constraints, T_slices]
            all_variables['Ps_over_slices'][batch_sample_idx] = all_Ps # [batch_size, n_slices, T_slices]
            all_variables['lambdas_over_slices'][batch_sample_idx] = all_lambdas # [batch_size, n_constraints, T_slices]
            all_variables['constraint_violations_over_slices'][batch_sample_idx] = all_constraint_violations

        # update init dual multipliers
        lambda_max_clamp = 1.
        lambda_lr = .5
        lambda_update_clamp = .1
        conservative_slack = 0.1
        dlambda = torch.clamp_(lambda_lr * (self.lambdas - all_variables['lambdas_over_slices'][:, :, -self.config.T_slices['train']:].mean(-1) - conservative_slack), min=-lambda_update_clamp, max=lambda_update_clamp)
        print(f'Epoch = {epoch}, dlambda = {dlambda}')
        self.lambdas = self.lambdas - dlambda
        self.lambdas.data.clamp_(min = 0, max = lambda_max_clamp)

        
        # metrics = SA_primal_optim_state(epoch = epoch,
        #                                 Lagrangian = None,
        #                                 obj = all_variables['obj'].detach().cpu().numpy(),
        #                                 constraints = all_variables['constraints'].detach().cpu().numpy(),
        #                                 pgrad_norm = None
        #                                 )

        # metrics_over_slices = SA_metrics_over_slices(epoch=epoch,
        #                                              obj_over_slices=all_variables['obj_over_slices'].detach().cpu().numpy(),
        #                                              constraints_over_slices=all_variables['constraints_over_slices'].detach().cpu().numpy(),
        #                                              Ps_over_slices=all_variables['Ps_over_slices'].detach().cpu().numpy(),
        #                                              lambdas_over_slices=all_variables['lambdas_over_slices'].detach().cpu().numpy(),
        #                                              constraint_violations_over_slices=all_variables['constraint_violations_over_slices'].detach().cpu().numpy()
        #                                              )
        
        # test_metrics = SA_test_state(epoch=epoch, metrics=metrics, metrics_over_slices=metrics_over_slices)
        
        return None, None


    def primal_step(self, epoch):
        '''
        Minimize the expected augmented Lagrangian to obtain the optimal state-augmented policy $\phi^\star$
        '''
        self.phase = 'train'
        n_constraints = len(self.constraint_fncs)
        n_slices = self.model.layers[-1].out_features # number of slice types that compete for bandwidth
        T_slices = self.config.T_slices[self.phase] # constraints and obj are averaged over T_slices slices.

        all_variables = defaultdict(list)
        all_variables['obj'] = torch.zeros((self.config.num_samples['train']), dtype=torch.float32).to(self.config.device)
        all_variables['constraints'] = torch.zeros((self.config.num_samples['train'], n_constraints), dtype=torch.float32).to(self.config.device)

        all_variables['obj_over_slices'] = torch.zeros((self.config.num_samples['train'], T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['constraints_over_slices'] = torch.zeros((self.config.num_samples['train'], n_constraints, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['Ps_over_slices'] = torch.zeros((self.config.num_samples['train'], n_slices, T_slices), dtype=torch.float32).to(self.config.device)

        # if self.lambda_samplers is not None:
        #     lambdas_all = torch.stack([sampler(size = (self.config.num_samples['train'],)) for sampler in self.lambda_samplers], dim = 1).to(self.config.device)
        # else:
        #     lambdas_all = torch.rand_like(all_variables['constraints']) # sample from U(0, 1) dist.

        self.model = self.model.to(self.config.device)
        self.model.train()

        # batches = torch.randperm(len(self.network_states[self.phase]))
        # batches = torch.split(batches, self.config.batch_size)
        for iter in range(self.config.n_primal_iters):
            batches = torch.randperm(len(self.network_states[self.phase]))
            batches = [torch.split(batches, self.config.batch_size)[0]]

            if self.lambda_samplers is not None:
                lambdas_all = torch.stack([sampler(size = (self.config.num_samples['train'],)) for sampler in self.lambda_samplers], dim = 1).to(self.config.device)
            else:
                lambdas_all = torch.rand_like(all_variables['constraints']) # sample from U(0, 1) dist.

            for batch_sample_idx in batches:
                self.model.zero_grad()
                batch_sample_idx = batch_sample_idx.tolist()
                lambdas = lambdas_all[batch_sample_idx].to(self.config.device)

                # all_Ps = []
                # all_obj_terms = []
                # all_constraint_terms = []
                with torch.set_grad_enabled(self.phase == 'train'):

                    # for logger in self.loggers:
                    #     logger(f'self.train(), epoch = {epoch}, subiter = {iter}, batch_sample_idx = {batch_sample_idx}, starting self.evaluate()')
                    obj_over_slices, constraints_over_slices, Ps_over_slices, _, _, _ = self.evaluate(lambdas = lambdas, Trange = trange(T_slices), batch_sample_idx=batch_sample_idx)
                    # for logger in self.loggers:
                    #     logger(f'self.train(), epoch = {epoch}, subiter = {iter}, batch_sample_idx = {batch_sample_idx}, finishing self.evaluate()')

                    all_Ps = torch.stack(Ps_over_slices, dim = -1) # [batch_size, 3, T_slices]
                    all_obj_terms = torch.stack(obj_over_slices, dim = -1) #[batch_size, T_slices]
                    all_constraint_terms = torch.stack(constraints_over_slices, dim = -1) #[batch_size, n_constraints, T_slices]

                    obj = torch.mean(all_obj_terms, dim = -1)
                    constraint_slacks = torch.mean(all_constraint_terms, dim = -1)
                    L = torch.mean(obj + (lambdas * constraint_slacks).sum(-1))
                    L.backward()

                    if self.config.pgrad_clipping_constant is not None:
                        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.pgrad_clipping_constant)  # Clip gradient norms

                    params = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                    pgrad_norm = np.sqrt(np.sum([p.grad.norm().item()**2 for p in params]))

                    if iter == self.config.n_primal_iters - 1:
                        all_variables['pgrad_norm'].append(pgrad_norm.item())
                        all_variables['Lagrangian'].append(L.item())
                        all_variables['obj'][batch_sample_idx] = obj # [batch_size]
                        all_variables['constraints'][batch_sample_idx] = constraint_slacks # [batch_size, n_constraints]
                        all_variables['obj_over_slices'][batch_sample_idx] = all_obj_terms # [batch_size, T_slices]
                        all_variables['constraints_over_slices'][batch_sample_idx] = all_constraint_terms # [batch_size, n_constraints, T_slices]
                        all_variables['Ps_over_slices'][batch_sample_idx] = all_Ps # [batch_size, n_slices, T_slices]

                    if self.primal_optim is not None:
                        self.primal_optim.step()
                        self.primal_optim.zero_grad()
                    else:
                        raise NotImplementedError
                    
        self.primal_lr_scheduler.step() if self.primal_lr_scheduler is not None else None

        all_variables['pgrad_norm'] = sum(all_variables['pgrad_norm']) / len(all_variables['pgrad_norm'])
        all_variables['Lagrangian'] = sum(all_variables['Lagrangian']) / len(all_variables['Lagrangian'])

        primal_optim_state = SA_primal_optim_state(epoch = epoch,
                                                    Lagrangian = all_variables['Lagrangian'],
                                                    obj = all_variables['obj'].detach().cpu().numpy(),
                                                    constraints = all_variables['constraints'].detach().cpu().numpy(),
                                                    pgrad_norm = all_variables['pgrad_norm']
                                                    )
        
        metrics_over_slices = SA_metrics_over_slices(epoch=epoch,
                                                     obj_over_slices=all_variables['obj_over_slices'].detach().cpu().numpy(),
                                                     constraints_over_slices=all_variables['constraints_over_slices'].detach().cpu().numpy(),
                                                     Ps_over_slices=all_variables['Ps_over_slices'].detach().cpu().numpy(),
                                                     lambdas_over_slices=lambdas.detach().cpu().numpy(),
                                                     constraint_violations_over_slices=None,
                                                     slicing_metrics_over_slices=None
                                                     )
        
        return primal_optim_state, metrics_over_slices
    


    def test(self, epoch, test_config):

        self.phase = 'test'
        n_constraints = len(self.constraint_fncs)
        n_slices = self.model.layers[-1].out_features # number of slice types that compete for bandwidth
        T_slices = self.config.T_slices[self.phase] # constraints and obj are averaged over T_slices slices.
        n_clients = self.network_states[self.phase][0].len

        all_variables = defaultdict(list)
        all_variables['obj'] = torch.zeros((self.config.num_samples['test']), dtype=torch.float32).to(self.config.device)
        all_variables['constraints'] = torch.zeros((self.config.num_samples['test'], n_constraints), dtype=torch.float32).to(self.config.device)

        all_variables['obj_over_slices'] = torch.zeros((self.config.num_samples['test'], T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['constraints_over_slices'] = torch.zeros((self.config.num_samples['test'], n_constraints, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['lambdas_over_slices'] = torch.zeros((self.config.num_samples['test'], n_constraints, T_slices), dtype=torch.float32).to(device='cpu')
        all_variables['Ps_over_slices'] = torch.zeros((self.config.num_samples['test'], n_slices, T_slices), dtype=torch.float32).to(self.config.device)
        # all_variables['constraint_violations_over_slices'] = torch.zeros((self.config.num_samples['test'], n_constraints, n_clients, T_slices), dtype=torch.bool).to(self.config.device)
        all_variables['constraint_violations_over_slices'] = torch.zeros((self.config.num_samples['test'], n_constraints, T_slices), dtype=torch.float32).to(self.config.device)

        # slicing metrics
        all_variables['slicing_metric'] = defaultdict(list)
        all_variables['slicing_metric']['throughput'] = torch.zeros((self.config.num_samples['test'], n_clients, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['slicing_metric']['rate'] = torch.zeros((self.config.num_samples['test'], n_clients, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['slicing_metric']['latency_avg'] = torch.zeros((self.config.num_samples['test'], n_clients, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['slicing_metric']['latency_quantile'] = torch.zeros((self.config.num_samples['test'], n_clients, T_slices), dtype=torch.float32).to(self.config.device)
        all_variables['slicing_metric']['slice_association'] = torch.zeros((self.config.num_samples['test'], n_clients, T_slices), dtype=torch.float32).to(self.config.device)

        if test_config.dual_test_init_strategy == 'zeros':
            lambdas_all = torch.zeros_like(all_variables['constraints']) # sample from U(0, 1) dist.
        elif test_config.dual_test_init_strategy == 'adaptive':
            lambdas_all = self.lambdas.detach().clone().to(self.config.device)
        else:
            raise NotImplementedError

        self.model = self.model.to(self.config.device)
        self.model.eval()

        batches = torch.randperm(len(self.network_states[self.phase]))
        batches = torch.split(batches, self.config.batch_size)

        for batch_sample_idx in batches:
            self.model.zero_grad()
            batch_sample_idx = batch_sample_idx.tolist()
            lambdas = lambdas_all[batch_sample_idx].to(self.config.device)

            # for logger in self.loggers:
            #     logger(f'self.test(), epoch = {epoch}, batch_sample_idx = {batch_sample_idx}, starting self.evaluate()')
            obj_over_slices, constraints_over_slices, Ps_over_slices, lambdas_over_slices, constraint_violations_over_slices, slicing_metrics_over_slices = self.evaluate(lambdas = lambdas, Trange = trange(T_slices), batch_sample_idx=batch_sample_idx, slicing_strategy = test_config.slicing_strategy)
            # for logger in self.loggers:
            #     logger(f'self.test(), epoch = {epoch}, batch_sample_idx = {batch_sample_idx}, finishing self.evaluate()')

            all_Ps = torch.stack(Ps_over_slices, dim = -1) # [batch_size, 3, T_slices]
            all_obj_terms = torch.stack(obj_over_slices, dim = -1) #[batch_size, T_slices]
            all_constraint_terms = torch.stack(constraints_over_slices, dim = -1) #[batch_size, n_constraints, T_slices]
            all_lambdas = torch.stack(lambdas_over_slices, dim = -1) # [batch_size, n_constraints, T_slices]
            all_constraint_violations = torch.stack(constraint_violations_over_slices, dim = -1) # [batch_size, n_constraints, T_slices, n_clients]

            all_slicing_metrics = defaultdict(list)
            for key in slicing_metrics_over_slices:
                all_slicing_metrics[key] = torch.stack(slicing_metrics_over_slices[key], dim = -1) # [batch_size, n, T_slices]

            obj = torch.mean(all_obj_terms, dim = -1)
            constraint_slacks = torch.mean(all_constraint_terms, dim = -1)

            all_variables['obj'][batch_sample_idx] = obj # [batch_size]
            all_variables['constraints'][batch_sample_idx] = constraint_slacks # [batch_size, n_constraints]
            all_variables['obj_over_slices'][batch_sample_idx] = all_obj_terms # [batch_size, T_slices]
            all_variables['constraints_over_slices'][batch_sample_idx] = all_constraint_terms # [batch_size, n_constraints, T_slices]
            all_variables['Ps_over_slices'][batch_sample_idx] = all_Ps # [batch_size, n_slices, T_slices]
            all_variables['lambdas_over_slices'][batch_sample_idx] = all_lambdas # [batch_size, n_constraints, T_slices]
            all_variables['constraint_violations_over_slices'][batch_sample_idx] = all_constraint_violations

            for key in slicing_metrics_over_slices:
                if key in all_variables['slicing_metric']:
                    all_variables['slicing_metric'][key][batch_sample_idx] = all_slicing_metrics[key]

        for key in all_variables['slicing_metric']:
            all_variables['slicing_metric'][key] = all_variables['slicing_metric'][key].detach().cpu().numpy()
        # # update init dual multipliers
        # lambda_max_clamp = 1.
        # lambda_lr = 1.
        # lambda_update_clamp = .1
        # conservative_slack = 0.1
        # dlambda = torch.clamp_(lambda_lr * (self.lambdas - all_variables['lambdas_over_slices'][:, :, 0:self.config.T_slices['train']].mean(-1) - conservative_slack), min=-lambda_update_clamp, max=lambda_update_clamp)
        # print(f'Epoch = {epoch}, dlambda = {dlambda}')
        # self.lambdas = self.lambdas - dlambda
        # self.lambdas.clamp_(min = 0, max = lambda_max_clamp)

        
        metrics = SA_primal_optim_state(epoch = epoch,
                                        Lagrangian = None,
                                        obj = all_variables['obj'].detach().cpu().numpy(),
                                        constraints = all_variables['constraints'].detach().cpu().numpy(),
                                        pgrad_norm = None
                                        )
        
        # all_variables['slicing_metric'] = defaultdict(list)
        # keys = list(all_variables.keys()).copy()
        # for key in keys:
        #     if len(all_variables['slicing_metric', key]):
        #         all_variables['slicing_metric'][key] = all_variables['slicing_metric', key].detach().clone().cpu().numpy()
        #         del all_variables[key]

        metrics_over_slices = SA_metrics_over_slices(epoch=epoch,
                                                     obj_over_slices=all_variables['obj_over_slices'].detach().cpu().numpy(),
                                                     constraints_over_slices=all_variables['constraints_over_slices'].detach().cpu().numpy(),
                                                     Ps_over_slices=all_variables['Ps_over_slices'].detach().cpu().numpy(),
                                                     lambdas_over_slices=all_variables['lambdas_over_slices'].detach().cpu().numpy(),
                                                     constraint_violations_over_slices=all_variables['constraint_violations_over_slices'].detach().cpu().numpy(),
                                                     slicing_metrics_over_slices=all_variables['slicing_metric']
                                                     )
        
        test_metrics = SA_test_state(epoch=epoch, metrics=metrics, metrics_over_slices=metrics_over_slices)
        
        return test_metrics
    

    def simulate_one_slicing_window(self, network, p, t):
        network.step(p, t)
        obj_term = self.obj_fnc(network.slicing_metrics[-1], slices = [client.slice for client in network.clients])
        constraint_term = [constraint_fnc(network.slicing_metrics[-1], slices = [client.slice for client in network.clients]) for constraint_fnc in self.constraint_fncs]
        constraint_violations_term = [[constraint_fnc(network.slicing_metrics[-1], slices = [client.slice]) > 1e-2 for client in network.clients if client.slice in constraint_fnc.eval_slices] for constraint_fnc in self.constraint_fncs]
        for term in constraint_violations_term:
            if len(term) == 0:
                term = 0
            else:
                term = sum(term) / len(term)

        return network, obj_term, constraint_term, constraint_violations_term
    

    def evaluate(self, lambdas, Trange, batch_sample_idx, slicing_strategy = 'state-augmented'):
        T_0 = self.config.T_0[self.phase]
        n_slices = self.model.layers[-1].out_features
        lr_dual = self.config.lr_dual[self.phase]
        with torch.set_grad_enabled(self.phase == 'train'):
            all_Ps = []
            all_obj_terms = []
            all_constraint_terms = []
            all_lambdas = []
            all_constraint_violations_terms = []

            metrics_type = ['throughput', 'rate', 'slice_rate', 'latency_quantile', 'latency_avg']
            all_metrics = defaultdict(list)
            for _ in metrics_type:
                all_metrics[_] = []
            # all_metrics = [[] for _ in range(len(metrics_type))]

            for t_idx, t in enumerate(Trange):
                channel_data = [self.network_states[self.phase][j].channel_data for j in batch_sample_idx]
                traffic_model = [[client.traffic_model for client in self.network_states[self.phase][j].clients] for j in batch_sample_idx]
                slice_features = self.feature_extractor(channel_data, traffic_model).to(self.config.device)

                if slicing_strategy == 'state-augmented':
                    p = self.model(slice_features, lambdas.detach()) # slicing decisions
                elif slicing_strategy == 'proportional':
                    p = torch.zeros((len(batch_sample_idx), n_slices)).to(self.config.device)
                    for idx, network in enumerate([self.network_states[self.phase][j] for j in batch_sample_idx]):
                        p[idx] = torch.tensor([len(network.slice(slice)) / (network.len - len(network.slice(Slice.IA))) for slice in Slice.list() if not slice == Slice.IA])
                elif slicing_strategy == 'uniform':
                    p = 1/n_slices * torch.ones((len(batch_sample_idx), n_slices)).to(self.config.device)
                elif slicing_strategy.startswith('mono-slicing'):
                    max_slice_width = 0.9
                    p = (1 - max_slice_width) / (n_slices - 1) * torch.ones((len(batch_sample_idx), n_slices)).to(self.config.device)
                    if slicing_strategy.endswith('HT'):
                        p[:, Slice.HT.value] = max_slice_width
                    elif slicing_strategy.endswith('LL'):
                        p[:, Slice.LL.value] = max_slice_width
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                obj_term = []
                constraint_term = []
                constraint_violations_term = []
                metrics_term = defaultdict(list)

                time_t = time.time()
                for idx, network in enumerate([self.network_states[self.phase][j] for j in batch_sample_idx]):
                    # Simulate one slicing window with given slicing decisions
                    for logger in self.loggers:
                        logger(f't_idx = {t_idx}, Network_idx = {network.network_idx}, starting network.step()')
                    network.step(p[idx], t)
                    for logger in self.loggers:
                        logger(f't_idx = {t_idx}, Network_idx = {network.network_idx}, finishing network.step()')

                    obj_term.append(self.obj_fnc(network.slicing_metrics[-1], slices = [client.slice for client in network.clients]))
                    constraint_term.append([constraint_fnc(network.slicing_metrics[-1], slices = [client.slice for client in network.clients]) for constraint_fnc in self.constraint_fncs])
                    # constraint_violations_term.append([[constraint_fnc(network.slicing_metrics[-1], slices = [client.slice]) > 0. for client in network.clients] for constraint_fnc in self.constraint_fncs])
                    constraint_violations = [[constraint_fnc(network.slicing_metrics[-1], slices = [client.slice]) > 1e-2 for client in network.clients if client.slice in constraint_fnc.eval_slices] for constraint_fnc in self.constraint_fncs]
                    for ii in range(len(constraint_violations)):
                        # print('term: ', constraint_violations[ii], 'sum: ', sum(constraint_violations[ii]))
                        if len(constraint_violations[ii]) == 0:
                            constraint_violations[ii] = 0
                        else:
                            constraint_violations[ii] = sum(constraint_violations[ii]) / len(constraint_violations[ii])
                    constraint_violations_term.append(constraint_violations)

                    # also save slicing metrics and slice associations
                    for key in all_metrics:
                        if key in network.slicing_metrics[-1]._asdict():
                            metrics_term[key].append(network.slicing_metrics[-1]._asdict()[key])

                    metrics_term['slice_association'].append(torch.tensor([client.slice.value for client in network.clients], dtype=torch.float32).to(self.config.device))
                
                # print('No parallelization it took: ', time.time() - time_t)
                # time_t = time.time()
                # results = Parallel(n_jobs=16)(delayed(self.simulate_one_slicing_window)(network, p[idx], t) for idx, network in enumerate([self.network_states[self.phase][j] for j in batch_sample_idx]))
                # for j, i in zip(batch_sample_idx, range(len(results))):
                #     self.network_states[self.phase][j] = results[i][0]
                # obj_term = [results[i][1] for i in range(len(results))]
                # constraint_term = [results[i][2] for i in range(len(results))]
                # constraint_violations_term = [results[i][3] for i in range(len(results))]
                # print('With parallelization it took: ', time.time() - time_t)
                obj_term = torch.stack(obj_term)
                constraint_term = torch.stack([torch.stack(term) for term in constraint_term])
                # constraint_violations_term = torch.stack([torch.stack([torch.stack(_) for _ in term]) for term in constraint_violations_term])
                constraint_violations_term = torch.stack([torch.stack(term) for term in constraint_violations_term])
                
                all_Ps.append(p)
                all_obj_terms.append(obj_term)
                all_constraint_terms.append(constraint_term)
                all_constraint_violations_terms.append(constraint_violations_term)
                
                for key in metrics_term:
                    metric = torch.stack(metrics_term[key])
                    all_metrics[key].append(metric)

                if self.phase == 'test':
                    if (t_idx + 1) % T_0 == 0:
                        constraint_slacks = torch.mean(torch.stack(all_constraint_terms[-T_0:], dim = -1), dim = -1)
                        dstep = lr_dual * constraint_slacks
                        lambdas += dstep
                        lambdas.data.clamp_(0)
                    all_lambdas.append(lambdas.detach().clone().cpu())

            return all_obj_terms, all_constraint_terms, all_Ps, all_lambdas, all_constraint_violations_terms, all_metrics
        

    def plot_train_primal_evolution_over_epochs(self, train_state_over_epochs, save_path = None, **kwargs):
        core.plot_utils.plot_train_primal_evolution_over_epochs(train_state_over_epochs, save_path = save_path, kwargs=kwargs)

    def plot_test_evolution_over_epochs(self, test_state_over_epochs, test_config_names, save_path = None, **kwargs):
        core.plot_utils.plot_test_evolution_over_epochs(self, test_state_over_epochs, test_config_names=test_config_names, save_path=save_path, kwargs=kwargs)

    def plot_test_evolution_over_slices(self, test_state_over_slices, test_config_names, save_path = None, network_idx = None, **kwargs):
        core.plot_utils.plot_test_evolution_over_slices(self, test_state_over_slices, test_config_names = test_config_names, save_path=save_path, network_idx=network_idx, kwargs=kwargs)

    def plot_traffic_evolution_over_slices(self, traffics, save_path = None, **kwargs):
        core.plot_utils.plot_traffic_evolution_over_slices(self, traffics=traffics, save_path=save_path, kwargs=kwargs)




                    




                    


        
        




