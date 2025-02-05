"""
Train the signaling model.
"""
from typing import Dict, List, Union
import time
import logging

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import LEMBAS.utilities as utils

LR_PARAMS = {'max_iter': 5000, 'learning_rate': 2e-3}
OTHER_PARAMS = {'batch_size': 8, 'noise_level': 10, 'gradient_noise_level': 1e-9}
REGULARIZATION_PARAMS = {'param_lambda_L2': 1e-6, 'output_bias_lambda_L2': 1e-6,
                         'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                         'uniform_min': 0,
                   'uniform_max': (1/1.2), 'spectral_loss_factor': 1e-5}
SPECTRAL_RADIUS_PARAMS = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS}

def split_data(X_in: torch.Tensor, 
               y_out: torch.Tensor, 
               train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None}, 
              seed: int = 888):
    """Splits the data into train, test, and validation.

    Parameters
    ----------
    X_in : torch.Tensor
        input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
    y_out : torch.Tensor
        output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    seed : int, optional
        seed value, by default 888
    """
    
    if not np.isclose(sum([v for v in train_split_frac.values() if v]), 1):
        raise ValueError('Train-test-validation split must sum to 1')
    
    if not train_split_frac['validation'] or train_split_frac['validation'] == 0:
        """X_train, X_test, y_train, y_test = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)"""
        # Naive split implementation
        X_train, X_test = train_test_split(  # Split X conditions
            X_in,
            train_size=train_split_frac['train'],
            random_state=seed
        )
        train_conditions = X_train.index.astype(str)
        test_conditions = X_test.index.astype(str)

        # Split the y_out to keep all time points for the respective conditions
        y_out = y_out.reset_index()
        y_out['Time'] = y_out['Drug_CL_Time'].str.split('_').str[-1]
        y_out['Drug_CL'] = y_out['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
        y_train = y_out[y_out['Drug_CL'].isin(train_conditions)]
        y_test = y_out[y_out['Drug_CL'].isin(test_conditions)]
        y_train = y_train.drop(columns=['Drug_CL_Time'])
        y_test = y_test.drop(columns=['Drug_CL_Time'])
        y_train['Drug_CL_Time'] = y_train['Drug_CL'] + '_' + y_train['Time']
        y_train = y_train.set_index('Drug_CL_Time').drop(columns=['Drug_CL', 'Time'])
        y_test['Drug_CL_Time'] = y_test['Drug_CL'] + '_' + y_test['Time']
        y_test = y_test.set_index('Drug_CL_Time').drop(columns=['Drug_CL', 'Time'])
        
        X_val, y_val = None, None
    else:
        X_train, _X, y_train, _y = train_test_split(X_in, 
                                                        y_out, 
                                                        train_size=train_split_frac['train'],
                                                        random_state=seed)
        X_test, X_val, y_test, y_val = train_test_split(_X, 
                                                    _y, 
                                                    train_size=train_split_frac['test']/(train_split_frac['test'] + train_split_frac['validation']),
                                                    random_state=seed)
    #print(y_train.head())
    return X_train, X_test, X_val, y_train, y_test, y_val

class ModelData(Dataset):
    def __init__(self, X_in, y_out, X_index, y_index):
        self.X_in = X_in
        self.y_out = y_out
        self.X_index = X_index
        self.y_index = y_index
        
        # Create a mapping from condition to the corresponding indices in y_out
        self.condition_to_indices = {}
        for condition in X_index:
            condition_str = str(condition)
            self.condition_to_indices[condition] = [i for i, y_idx in enumerate(y_index) if y_idx.startswith(condition_str + '_')]

    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.size(0)

    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        condition = self.X_index[idx]
        X_item = self.X_in[idx]
        # Get all corresponding Y items for the condition
        Y_indices = self.condition_to_indices[condition]
        Y_items = self.y_out[Y_indices]
        return X_item, Y_items
    """def __init__(self, X_in, y_out):
        self.X_in = X_in
        self.y_out = y_out
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.X_in.shape[0]
    
    def __getitem__(self, idx: int):
        "Returns one sample of data, data and label (X, y)."
        return self.X_in[idx, :], self.y_out[idx, :]"""

def train_signaling_model(mod,  
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch : int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None,
                          train_split_frac: Dict = {'train': 0.8, 'test': 0.2, 'validation': None},
                          train_seed: int = None,
                         verbose: bool = True, 
                         break_nan: bool = True):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    optimizer : torch.optim.adam.Adam
        optimizer to use during training
    loss_fn : torch.nn.modules.loss.MSELoss
        loss function to use during training
    reset_epoch : int, optional
        number of epochs upon which to reset the optimizer state, by default 200
    hyper_params : Dict[str, Union[int, float]], optional
        various hyper parameter inputs for training
            - 'max_iter' : the number of epochs, by default 5000
            - 'learning_rate' : the starting learning rate, by default 2e-3
            - 'batch_size' : number of samples per batch, by default 8
            - 'noise_level' : noise added to signaling network input, by default 10. Set to 0 for no noise. Makes model more robust. 
            - 'gradient_noise_level' : noise added to gradient after backward pass. Makes model more robust. 
            - 'reset_epoch' : number of epochs upon which to reset the optimizer state, by default 200
            - 'param_lambda_L2' : L2 regularization penalty term for most of the model weights and biases
            - 'output_bias_lambda_L2' : L2 regularization penalty term for ProjectOutput layer bias
            - 'moa_lambda_L1' : L1 regularization penalty term for incorrect interaction mechanism of action (inhibiting/stimulating)
            - 'ligand_lambda_L2' : L2 regularization penalty term for ligand biases
            - 'uniform_lambda_L2' : L2 regularization penalty term for 
            - 'uniform_max' : 
            - 'spectral_loss_factor' : regularization penalty term for 
            - 'n_probes_spectral' : 
            - 'power_steps_spectral' : 
            - 'subset_n_spectral' : 
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train, test and split, by default 0.8, 0.2, and 0 respectively
    train_seed : int, optional
        seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
        with the same mod.seed, e.g., for cross-validation
    verbose : bool, optional
        whether to print various progress stats across training epochs
    break_nan : bool, optional
        whether to break the training loop if params containt nan


    Returns
    -------
    mod : SignalingModel
        a copy of the input model with trained parameters
    cur_loss : List[float], optional
        a list of the loss (excluding regularizations) across training iterations
    cur_eig : List[float], optional
        a list of the spectral_radius across training iterations
    mean_loss : torch.Tensor
        mean TF activity loss across samples (independent of training)
    X_train : torch.Tensor
        the train split of the input data
    X_test : torch.Tensor
        the test split of the input data
    X_val : torch.Tensor
        the validation split of the input data
    y_train : torch.Tensor
        the train split of the output data
    y_test : torch.Tensor
        the test split of the output data
    y_val : torch.Tensor
        the validation split of the output data
    """
    if not hyper_params:
        hyper_params = HYPER_PARAMS.copy()
    else:
        hyper_params = {k: v for k,v in {**HYPER_PARAMS, **hyper_params}.items() if k in HYPER_PARAMS} # give user input priority
    
    stats = utils.initialize_progress(hyper_params['max_iter'])

    mod = mod.copy() # do not overwrite input
    optimizer = optimizer(mod.parameters(), lr=1, weight_decay=0)
    reset_state = optimizer.state.copy()

    #X_in = mod.df_to_tensor(mod.X_in)
    #y_out = mod.df_to_tensor(mod.y_out)
    #mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean TF (across samples) loss
    X_in = mod.X_in
    y_out = mod.y_out

    # set up data objects
    
    if not train_seed:
        train_seed = mod.seed
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X_in, y_out, train_split_frac, train_seed)
    
    # Store the indexes for batch matching
    X_train_index = X_train.index.tolist()
    y_train_index = y_train.index.tolist()
    
    X_in = mod.df_to_tensor(X_in)
    y_out = mod.df_to_tensor(y_out)
    X_train = mod.df_to_tensor(X_train)
    X_test = mod.df_to_tensor(X_test)
    y_train = mod.df_to_tensor(y_train)
    y_test = mod.df_to_tensor(y_test)
    mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean TF (across samples) loss
    
    train_data = ModelData(X_train.to('cpu'), y_train.to('cpu'), X_train_index, y_train_index)
    
    if mod.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    # if n_cores != 0:
    #     n_cores_train = min(n_cores, hyper_params['batch_size'])
    # else:
    #     n_cores_train = n_cores
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=hyper_params['batch_size'],
                                  # num_workers=n_cores_train,
                                  drop_last = False,
                                  pin_memory = pin_memory,
                                  shuffle=True) 
    start_time = time.time()
    # begin iteration 
    mod.signaling_network.force_sparcity()
    for e in range(hyper_params['max_iter']):
        # set learning rate
        cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height = hyper_params['learning_rate'],
                              start_height=hyper_params['learning_rate']/10, end_height=1e-6, peak = 1000)
        optimizer.param_groups[0]['lr'] = cur_lr
        
        cur_loss = []
        cur_eig = []
        
        # iterate through batches
        if mod.seed:
            utils.set_seeds(mod.seed + e)
        for batch, (X_in_, y_out_) in enumerate(train_dataloader):
            mod.train()
            optimizer.zero_grad()

            X_in_, y_out_ = X_in_.to(mod.device), y_out_.to(mod.device)
            
            # forward pass
            X_full = mod.input_layer(X_in_) # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
            Y_full, Y_fullFull = mod.signaling_network(X_full) # train signaling network weights
            Y_hat = mod.output_layer(Y_full)
            
            time_points = [int(idx.rsplit('_', 1)[-1]) for idx in y_train_index]
            seen = set()
            unique_time_points = [x for x in time_points if not (x in seen or seen.add(x))]
            Y_subsampled = Y_fullFull[:, unique_time_points, :]
            
            # get prediction loss
            fit_loss = loss_fn(y_out_, Y_subsampled)

            # get regularization losses
            sign_reg = mod.signaling_network.sign_regularization(lambda_L1 = hyper_params['moa_lambda_L1']) # incorrect MoA
            ligand_reg = mod.ligand_regularization(lambda_L2 = hyper_params['ligand_lambda_L2']) # ligand biases
            stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(Y_full = Y_full.detach(), spectral_loss_factor = hyper_params['spectral_loss_factor'],
                                                                                subset_n = hyper_params['subset_n_spectral'], n_probes = hyper_params['n_probes_spectral'], 
                                                                                power_steps = hyper_params['power_steps_spectral'])
            uniform_reg = mod.uniform_regularization(lambda_L2 = hyper_params['uniform_lambda_L2']*cur_lr, Y_full = Y_full, 
                                                     target_min = hyper_params['uniform_min'], target_max = hyper_params['uniform_max']) # uniform distribution
            
            param_reg = mod.L2_reg(lambda_L2 = hyper_params['param_lambda_L2']) # all model weights and signaling network biases
            
            total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg
    
            # gradient
            total_loss.backward()
            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_level'])
            optimizer.step()
            mod.signaling_network.force_sparcity()
            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())
    
        stats = utils.update_progress(stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                     n_sign_mismatches = mod.signaling_network.count_sign_mismatch())
        
        if break_nan and (e % (hyper_params['max_iter']/100) == 0):
            param_names = []
            for name, param in mod.named_parameters():
                if torch.isnan(param).any():
                    param_names.append(name)
            if len(param_names) > 0:
                log_error = 'NaN values found in model parameters at epoch {}'.format(e)
                log_error += ' for layers ' + ', '.join(param_names)
                logging.error(log_error)
                raise ValueError('NaN values found in model parameters at epoch {}'.format(e))
        
        if verbose and e % 250 == 0:
            utils.print_stats(stats, iter = e)
        
        if np.logical_and(e % reset_epoch == 0, e>0):
            optimizer.state = reset_state.copy()
      

    if verbose:
        mins, secs = divmod(time.time() - start_time, 60)
        print("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

    return mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_val, y_train, y_test, y_val