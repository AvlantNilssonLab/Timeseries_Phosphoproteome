"""
Train the signaling model.
"""
from typing import Dict, List, Union, Literal
import time
import logging

import numpy as np
import pandas as pd
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
MODULE_PARAMS = {'use_time': True}
HYPER_PARAMS = {**LR_PARAMS, **OTHER_PARAMS, **REGULARIZATION_PARAMS, **SPECTRAL_RADIUS_PARAMS, **MODULE_PARAMS}

def split_data(X_in: torch.Tensor, 
               y_out: torch.Tensor, 
               train_split_frac: Dict = {'train': 0.8, 'test': 0.2}, 
               seed: int = 888,
               split_by: Literal['time', 'condition'] = 'time'):
    """Splits the data into train, test, and validation. It keeps specific time points for training and others for testing if split_by == 'time'. 
    If split_by == 'condition' it keeps all time points of specific conditions in the training while all other conditions for testing.

    Parameters
    ----------
    X_in : torch.Tensor
        input ligand concentrations. Index represents samples and columns represent a ligand. Values represent amount of ligand introduced (e.g., concentration). 
    y_out : torch.Tensor
        output TF activities. Index represents samples and columns represent TFs. Values represent activity of the TF.
    train_split_frac : Dict, optional
        fraction of samples to be assigned to each of train and test, by default 0.8 and 0.2 respectively
    seed : int, optional
        seed value, by default 888
    """
    
    if not np.isclose(sum([v for v in train_split_frac.values() if v]), 1):
        raise ValueError('Train-test-validation split must sum to 1')
    
    if split_by == 'time':
        # Time split
        y_out = y_out.reset_index()
        y_out['Time'] = y_out['Drug_CL_Time'].str.split('_').str[-1].astype(int)
        y_out['Drug_CL'] = y_out['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
        unique_time_points = y_out['Time'].unique()
        
        # Ensure the first and last time points are always in the training set
        first_time_point = unique_time_points.min()
        last_time_point = unique_time_points.max()
        remaining_time_points = np.setdiff1d(unique_time_points, [first_time_point, last_time_point])
        
        # Determine number of time points for splits and randomly sample them
        n_train_time_points = int(round(len(unique_time_points) * train_split_frac['train'])) - 1
        np.random.seed(seed)
        train_time_points = np.random.choice(remaining_time_points, n_train_time_points, replace=False)
        train_time_points = np.sort((np.concatenate(([first_time_point], train_time_points, [last_time_point]))))
        test_time_points = np.sort(np.setdiff1d(unique_time_points, train_time_points))
        print(f'Time points selected for training set: {train_time_points}')
        
        # Split the data based on the selected time points
        y_train = y_out[y_out['Time'].isin(train_time_points)]
        y_test = y_out[y_out['Time'].isin(test_time_points)]
        y_train = y_train.sort_values(by=['Time', 'Drug_CL'])
        y_test = y_test.sort_values(by=['Time', 'Drug_CL'])
        
        y_train = y_train.set_index('Drug_CL_Time').drop(columns=['Time', 'Drug_CL'])
        y_test = y_test.set_index('Drug_CL_Time').drop(columns=['Time', 'Drug_CL'])
        
        # Split X_in based on the conditions in y_train and y_test
        train_conditions = y_train.index.str.rsplit('_', n=1).str[0].unique()
        test_conditions = y_test.index.str.rsplit('_', n=1).str[0].unique()
        X_train = X_in.loc[train_conditions]
        X_test = X_in.loc[test_conditions]
        
        return X_train, X_test, y_train, y_test, train_time_points.tolist(), test_time_points.tolist()
        
    else:
        if train_split_frac['train'] == 1:
            X_train = X_in
            y_train = y_out
            X_test = None
            y_test = None
        else:
            # Condition split
            X_train, X_test = train_test_split(  # Split X conditions
                X_in,
                train_size=train_split_frac['train'],
                random_state=seed
            )
            train_conditions = X_train.index.astype(str)
            test_conditions = X_test.index.astype(str)
            
            # Split the y_out to keep all time points for the respective conditions
            y_out = y_out.reset_index()
            y_out['Time'] = y_out['Drug_CL_Time'].str.split('_').str[-1].astype(int)
            y_out['Drug_CL'] = y_out['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
            y_train = y_out[y_out['Drug_CL'].isin(train_conditions)]
            y_test = y_out[y_out['Drug_CL'].isin(test_conditions)]
            y_train = y_train.drop(columns=['Drug_CL_Time'])
            y_test = y_test.drop(columns=['Drug_CL_Time'])
            
            y_train = y_train.sort_values(by=['Time', 'Drug_CL'])
            y_test = y_test.sort_values(by=['Time', 'Drug_CL'])
            
            # Map y index to X
            X_train = X_in.loc[y_train['Drug_CL'].unique()]
            X_test = X_in.loc[y_test['Drug_CL'].unique()]
            
            y_train['Drug_CL_Time'] = y_train['Drug_CL'] + '_' + y_train['Time'].astype(str)
            y_train = y_train.set_index('Drug_CL_Time').drop(columns=['Drug_CL', 'Time'])
            y_test['Drug_CL_Time'] = y_test['Drug_CL'] + '_' + y_test['Time'].astype(str)
            y_test = y_test.set_index('Drug_CL_Time').drop(columns=['Drug_CL', 'Time'])
            
            return X_train, X_test, y_train, y_test
    
    return X_train, X_test, y_train, y_test


class ModelData(Dataset):
    def __init__(self, X_in, y_out, X_cell, mask, X_index, y_index):
        self.X_in = X_in
        self.y_out = y_out
        self.X_cell = X_cell
        self.mask = mask
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
        X_cell_item = self.X_cell[idx]
        # Get all corresponding Y items for the condition
        Y_indices = self.condition_to_indices[condition]
        Y_items = self.y_out[Y_indices]
        mask_items = self.mask[Y_indices]
        
        return X_item, X_cell_item, Y_items, mask_items


def soft_index(Y: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Parametes
    ----------
    Y : torch.Tensor
        the hidden state outputs of your RNN
    indices : torch.Tensor
        tensor of shape (K,) with continuous indices in [0, L-1]
    
    Returns
    ----------
    Y_selected : torch.Tensor
        computed by linearly interpolating along the time dimension.
    """
    batch, L, feat = Y.shape
    Y_cpu = Y.to('cpu')
    
    # For each continuous index, compute floor and ceil indices and weights.
    floor_idx = torch.floor(indices).long()  # shape (K,)
    ceil_idx = torch.clamp(floor_idx + 1, max=L-1)  # shape (K,)
    weight = (indices - floor_idx.float()).view(1, -1, 1)  # shape (1, K, 1)
    
    # Expand floor and ceil indices to have shape (batch, K, 1) for indexing.
    floor_idx_full = floor_idx.view(1, -1, 1).expand(batch, -1, feat)
    ceil_idx_full = ceil_idx.view(1, -1, 1).expand(batch, -1, feat)

    # Gather the corresponding hidden state outputs.
    # Y has shape (batch, L, features); we need to index along dimension 1.
    Y_floor = torch.gather(Y_cpu, 1, floor_idx_full)  # (batch, K, feat)
    Y_ceil = torch.gather(Y_cpu, 1, ceil_idx_full)      # (batch, K, feat)
    
    # Perform linear interpolation:
    Y_selected = (1 - weight) * Y_floor + weight * Y_ceil
    #time_idx = torch.round((floor_idx + ceil_idx)/2)
    
    return Y_selected.to(Y.device), floor_idx_full, ceil_idx_full, weight

def add_input_noise(X, noise_scale):
    """
    Adds Gaussian noise to the input tensor.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor.
    noise_scale : float
        Scale factor used to multiply the noise.

    Returns
    -------
    torch.Tensor
        Noisy version of X.
    """
    return X + noise_scale * torch.randn_like(X)

def train_signaling_model(mod,
                          net: pd.DataFrame,
                          optimizer: torch.optim, 
                          loss_fn: torch.nn.modules.loss,
                          reset_epoch : int = 200,
                          hyper_params: Dict[str, Union[int, float]] = None,
                          train_split_frac: Dict = {'train': 0.8, 'test': 0.2},
                          train_seed: int = None,
                          verbose: bool = True, 
                          break_nan: bool = True,
                          split_by: Literal['time', 'condition'] = 'time',
                          noise_scale: float = 0):
    """Trains the signaling model

    Parameters
    ----------
    mod : SignalingModel
        initialized signaling model. Suggested to also run `mod.signaling_network.prescale_weights` prior to training
    net: pd.DataFrame
            signaling network adjacency list with the following columns:
                - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0). Exclude non-interacting (0)
                nodes. 
                - `source_label`: source node column name
                - `target_label`: target node column name
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
        fraction of samples to be assigned to each of train and test, by default 0.8 and 0.2 respectively
    train_seed : int, optional
        seed value, by default mod.seed. By explicitly making this an argument, it allows different train-test splits even 
        with the same mod.seed, e.g., for cross-validation
    verbose : bool, optional
        whether to print various progress stats across training epochs
    break_nan : bool, optional
        whether to break the training loop if params contain nan
    split_by : Literal['time', 'condition'], optional
        criterion to split the data, by default 'time'
    noise_scale : float, optional
        scale factor used to multiply the noise, by default 0

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
    #optimizer = optimizer(mod.parameters(), lr=1, weight_decay=0)
    lr_factor = 10
    base_lr = hyper_params['learning_rate']
    time_layer_lr = base_lr * lr_factor  # Higher learning rate for the time-mapping layer

    # Initialize the optimizer with parameter groups
    optimizer = torch.optim.Adam([
        {'params': mod.input_layer.parameters(), 'lr': base_lr},
        {'params': mod.signaling_network.parameters(), 'lr': base_lr},
        {'params': mod.output_layer.parameters(), 'lr': base_lr},
        {'params': mod.nodes_sites_layer.parameters(), 'lr': base_lr},
        {'params': mod.time_layer.parameters(), 'lr': time_layer_lr}
    ])
    reset_state = optimizer.state.copy()

    #X_in = mod.df_to_tensor(mod.X_in)
    #y_out = mod.df_to_tensor(mod.y_out)
    #mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean TF (across samples) loss
    X_in = mod.X_in
    y_out = mod.y_out
    X_cell = mod.X_cell
    nodes_sites_map = mod.nodes_sites_map
    
    # identify genes to remove from prediction (all PKN nodes) to calculate loss
    node_labels = sorted(pd.concat([net['source'], net['target']]).unique())
    missing_labels = [label for label in node_labels if label not in list(nodes_sites_map.columns)]  # Identify missing genes
    missing_indexes = [node_labels.index(label) for label in missing_labels]  

    # set up data objects
    if not train_seed:
        train_seed = mod.seed
    
    if split_by == 'time':
        X_train, X_test, y_train, y_test, train_time_points, test_time_points = split_data(X_in, y_out, train_split_frac, train_seed, split_by)
    else:
        X_train, X_test, y_train, y_test = split_data(X_in, y_out, train_split_frac, train_seed, split_by)
    
    # Store the indexes for batch matching
    X_train_index = X_train.index.tolist()
    X_test_index = X_test.index.tolist() if X_test is not None else None
    y_train_index = y_train.index.tolist()
    X_in = mod.df_to_tensor(X_in)
    y_out = mod.df_to_tensor(y_out)
    X_train = mod.df_to_tensor(X_train)
    X_test = mod.df_to_tensor(X_test) if X_test is not None else None
    y_train = mod.df_to_tensor(y_train)
    y_test = mod.df_to_tensor(y_test) if y_test is not None else None
    
    # Split X_cell if condition-based testing
    if split_by == 'condition':
        X_cell_train = X_cell.loc[X_train_index]
        X_cell_test = X_cell.loc[X_test_index] if X_test is not None else None
        X_cell_train = mod.df_to_tensor(X_cell_train)
        X_cell_test = mod.df_to_tensor(X_cell_test) if X_cell_test is not None else None
    else:
        X_cell_train = mod.df_to_tensor(X_cell)
        X_cell_test = mod.df_to_tensor(X_cell)
    
    mean_loss = loss_fn(torch.mean(y_out, dim=0) * torch.ones(y_out.shape, device = y_out.device), y_out) # mean (across samples) loss
    
    # Create NaN mask for y_train
    mask = ~torch.isnan(y_train)
    
    train_data = ModelData(X_train.to('cpu'), y_train.to('cpu'), X_cell_train.to('cpu'), mask.to('cpu'), X_train_index, y_train_index)
    
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
        # set learning rate either as a variable or constant
        if hyper_params.get("variable_lr", True):
            cur_lr = utils.get_lr(
                e,
                hyper_params['max_iter'],
                max_height=hyper_params['learning_rate'],
                start_height=hyper_params['learning_rate'] / 10,
                end_height=1e-6,
                peak=1000
            )
        else:
            cur_lr = hyper_params['learning_rate']
        optimizer.param_groups[0]['lr'] = cur_lr
        for param_group in optimizer.param_groups:
            if param_group['lr'] == time_layer_lr:
                param_group['lr'] = cur_lr * lr_factor  # Adjust time_layer learning rate
            else:
                param_group['lr'] = cur_lr
            
        cur_loss = []
        cur_eig = []
        
        # iterate through batches
        if mod.seed:
            utils.set_seeds(mod.seed + e)
        for batch, (X_in_, X_cell_, y_out_, mask_) in enumerate(train_dataloader):
            mod.train()
            optimizer.zero_grad()

            X_in_, X_cell_, y_out_, mask_ = X_in_.to(mod.device), X_cell_.to(mod.device), y_out_.to(mod.device), mask_.to(mod.device)

            # forward pass
            X_full = mod.input_layer(X_in_) # transform to full network with ligand input concentrations
            utils.set_seeds(mod.seed + mod._gradient_seed_counter)
            network_noise = torch.randn(X_full.shape, device = X_full.device)
            X_full = X_full + (hyper_params['noise_level'] * cur_lr * network_noise) # randomly add noise to signaling network input, makes model more robust
            Y_full, Y_fullFull = mod.signaling_network(X_full, X_cell_) # train signaling network weights
            
            # Subsample Y_subsampled 3rd dimension (signaling nodes) to calculate loss before map the phosphosites
            mask = torch.tensor([i not in missing_indexes for i in range(len(node_labels))])
            Y_fullFull = Y_fullFull[:, :, mask]
            Y_fullFull = mod.nodes_sites_layer(Y_fullFull)
            
            time_map = mod.time_layer()

            #Y_hat = mod.output_layer(Y_full)
            
            # Subsample Y_subsampled 2nd dimension (time points) to calculate loss
            time_points = [int(idx.rsplit('_', 1)[-1]) for idx in y_train_index]
            seen = set()
            unique_time_points = [x for x in time_points if not (x in seen or seen.add(x))]
            
            use_time = hyper_params['use_time']
            if use_time:
                Y_subsampled, floor_idx_full, ceil_idx_full, weight = soft_index(Y_fullFull, time_map)
            else:
                Y_subsampled = Y_fullFull[:, unique_time_points, :]
            
            # Mask NaN with 0 to skip loss calculation
            y_out_.masked_fill_(~mask_, 0.0)
            Y_subsampled.masked_fill_(~mask_, 0.0)
            
            y_out_noise = add_input_noise(y_out_, noise_scale)  # Add noise to the output for robustness and overfitting prevention
            
            # get prediction loss
            fit_loss = loss_fn(y_out_noise, Y_subsampled)
            
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
            #mod.time_layer.print_gradients()
            #print("---------")
            mod.add_gradient_noise(noise_level = hyper_params['gradient_noise_level'])
            optimizer.step()
            mod.signaling_network.force_sparcity()
            # store
            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())

        if use_time:
            # Approximate index for monitoring
            idx_mon = (1 - weight) * floor_idx_full + weight * ceil_idx_full
            idx_mon = idx_mon[0, :, 0].tolist()
        else:
            idx_mon = None
            floor_idx_full = None
            ceil_idx_full = None
            weight = None
        
        stats = utils.update_progress(stats, iter = e, loss = cur_loss, eig = cur_eig, learning_rate = cur_lr, 
                                     n_sign_mismatches = mod.signaling_network.count_sign_mismatch(), idx_mon = idx_mon)
        
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

    if split_by == 'time':
        return mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_train_index, y_train, y_test, y_train_index, X_cell_train, X_cell_test, train_time_points, test_time_points, missing_indexes, floor_idx_full, ceil_idx_full, weight
    else:
        return mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_train_index, y_train, y_test, y_train_index, X_cell_train, X_cell_test, missing_indexes, floor_idx_full, ceil_idx_full, weight