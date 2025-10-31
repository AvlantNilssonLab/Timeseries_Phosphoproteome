import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import time
import copy
import torch
import torch.nn as nn
import networkx as nx
import plotnine as p9
from plotnine.stats.stat_summary import mean_cl_boot
import sys

lem_bas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "LEMBAS"))
if lem_bas_dir not in sys.path:
    sys.path.insert(0, lem_bas_dir)
    
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io


def subsample_Y(Y, floor_idx, ceil_idx, weight):
    """
    Subsample Y using the provided indices and weights.

    Parameters
    ----------
    Y : torch.Tensor
        The tensor to be subsampled.
    floor_idx : torch.Tensor
        The floor indices for subsampling.
    ceil_idx : torch.Tensor
        The ceil indices for subsampling.
    weight : torch.Tensor
        The interpolation weights.

    Returns
    -------
    Y_subsampled : torch.Tensor
        The subsampled tensor.
    """
    Y = Y.to('cpu')
    
    # Reduce dimensions of indices
    floor_idx_reduced = floor_idx[0, :, 0]
    ceil_idx_reduced = ceil_idx[0, :, 0]
    
    # Gather the corresponding hidden state outputs
    Y_floor = Y[:, floor_idx_reduced, :]  # (batch, K, feat)
    Y_ceil = Y[:, ceil_idx_reduced, :]  # (batch, K, feat)
    
    # Perform linear interpolation
    Y_subsampled = (1 - weight) * Y_floor + weight * Y_ceil
    
    return Y_subsampled


def identify_controls(Y_sub, Y_act, y_index_CL, control_drug='D1'):
    """
    From your TRAINING 3D tensors, find the D1 control curve for each cell-line.

    Parameters
    ----------
    Y_sub : np.ndarray or torch.Tensor, shape (T, N, P)
        Subsampled tensor (e.g. model predictions).
    Y_act : np.ndarray or torch.Tensor, shape (T, N, P)
        Actual measurements tensor.
    y_index_CL : list of str, length N
        Entries like "D1_A549", "D2_MCF7", …
    control_drug : str
        Which drug to treat as control (default "D1").

    Returns
    -------
    control_curves : dict
        Mapping cell-line → dict with keys 'sub' and 'act', each an array/tensor of shape (T, P).
    """
    T, N, P = Y_sub.shape
    assert Y_act.shape == (T, N, P)
    assert len(y_index_CL) == N

    control_curves = {}
    for idx, entry in enumerate(y_index_CL):
        drug, cl = entry.split('_')
        if drug == control_drug:
            if cl in control_curves:
                print(f"⚠️ Multiple {control_drug} entries for '{cl}'—"
                      f"keeping first at idx={control_curves[cl]['idx']}, skipping idx={idx}")
            else:
                # pick off the time-course slice
                sub_slice = Y_sub[:, idx, :]
                act_slice = Y_act[:, idx, :]
                # clone for torch, copy for numpy
                sub_ctrl = sub_slice.clone() if hasattr(sub_slice, 'clone') else sub_slice.copy()
                act_ctrl = act_slice.clone() if hasattr(act_slice, 'clone') else act_slice.copy()

                control_curves[cl] = {
                    'sub': sub_ctrl,
                    'act': act_ctrl,
                    'idx': idx
                }
                print(f"✔️ Found control for '{cl}' at index {idx}")

    all_cls = set(e.split('_')[1] for e in y_index_CL)
    missing = all_cls - set(control_curves.keys())
    if missing:
        print(f"⚠️ No control ({control_drug}) found for cell lines: {sorted(missing)}")
    else:
        print(f"✅ Controls identified for all {len(all_cls)} cell lines.")
    return control_curves


def subtract_controls(Y_sub, Y_act, y_index_CL, control_curves, control_drug='D1'):
    """
    Subtracts previously-identified controls from ANY 3D tensor.

    Parameters
    ----------
    Y_sub : np.ndarray or torch.Tensor, shape (T, N, P)
        Tensor to adjust (e.g. predictions).
    Y_act : np.ndarray or torch.Tensor, shape (T, N, P)
        Tensor to adjust (e.g. actuals).
    y_index_CL : list of str, length N
        Entries like "D1_A549", "D2_MCF7", …
    control_curves : dict
        As returned by `identify_controls`, mapping cell-line → {'sub':…, 'act':…, 'idx':…}.
    control_drug : str
        Which drug was used as control (default 'D1').

    Returns
    -------
    Y_sub_adj, Y_act_adj : np.ndarray or torch.Tensor
        Baseline-corrected tensors (same type & shape as inputs).
    """
    T, N, P = Y_sub.shape
    assert Y_act.shape == (T, N, P)
    assert len(y_index_CL) == N

    # allocate the outputs in the same framework
    if hasattr(Y_sub, 'clone'):
        Y_sub_adj = Y_sub.clone()
        Y_act_adj = Y_act.clone()
    else:
        Y_sub_adj = np.empty_like(Y_sub)
        Y_act_adj = np.empty_like(Y_act)

    for idx, entry in enumerate(y_index_CL):
        drug, cl = entry.split('_')
        ctr = control_curves.get(cl)

        # if sample is DMSO don’t subtract anything
        if drug == 'DMSO':
            Y_sub_adj[:, idx, :] = Y_sub[:, idx, :]
            Y_act_adj[:, idx, :] = Y_act[:, idx, :]
            continue

        if ctr is None:
            Y_sub_adj[:, idx, :] = Y_sub[:, idx, :]
            Y_act_adj[:, idx, :] = Y_act[:, idx, :]

        else:
            if drug == control_drug:
                zero = (ctr['sub'] * 0) if hasattr(ctr['sub'], 'clone') else np.zeros_like(ctr['sub'])
                Y_sub_adj[:, idx, :] = zero
                Y_act_adj[:, idx, :] = zero
            else:
                Y_sub_adj[:, idx, :] = Y_sub[:, idx, :] - ctr['sub']
                Y_act_adj[:, idx, :] = Y_act[:, idx, :] - ctr['act']

    print("✅ Subtraction complete.")
    return Y_sub_adj, Y_act_adj


def train_cv(mod, net, hyper_params, n_cv = 5, config = None, cv_results = []):
    """
    Train a model using 5-fold cross-validation.
    
    Parameters
    ----------
    mod : SignalingModel
        The model to be trained.
    net : pd.DataFrame
        The prior knowledge signaling network.
    hyper_params : dict
        The hyperparameters to be used for training.
    n_cv : int
        The number of cross-validation folds.
    config : str
        The configuration name for the current run.
    cv_results : list
        A list to store the results of each fold.
    
    Returns
    -------
    cv_results : list
        The results of the cross-validation.
    """
    # Prepare 5xFolds
    indices = np.array(mod.X_in.index)
    d1_mask = np.char.startswith(indices.astype(str), "D1_")
    d1_idx = np.where(d1_mask)[0]
    rest_idx  = indices[~d1_mask]
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=49)

    for fold, (train_idx, test_idx) in enumerate(kf.split(rest_idx)):
        print(f"Processing fold {fold+1}...")
        # Reset model setup for each fold
        mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
        mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam

        train_idx = np.concatenate([train_idx, d1_idx])  # add D1 to training set
        print(f"Train indices: {train_idx}, Test indices: {test_idx}")
        mod_fold = copy.deepcopy(mod)
        mod_fold.X_in = mod.X_in.loc[indices[train_idx]]
        mod_fold.X_cell = mod.X_cell.loc[indices[train_idx]]
        
        y_out_proc = mod.y_out.reset_index()
        y_out_proc['Time'] = y_out_proc['Drug_CL_Time'].str.split('_').str[-1].astype(int)
        y_out_proc['Drug_CL'] = y_out_proc['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
        y_train_proc = y_out_proc[y_out_proc['Drug_CL'].isin(indices[train_idx])]
        y_test_proc = y_out_proc[y_out_proc['Drug_CL'].isin(indices[test_idx])]
        y_train_proc = y_train_proc.drop(columns=['Drug_CL_Time']).sort_values(by=['Time', 'Drug_CL'])
        y_test_proc = y_test_proc.drop(columns=['Drug_CL_Time']).sort_values(by=['Time', 'Drug_CL'])
        
        mod_fold.X_in = mod_fold.X_in.loc[y_train_proc['Drug_CL'].unique()]
        X_test =  mod.X_in.loc[indices[test_idx]]
        X_test = X_test.loc[y_test_proc['Drug_CL'].unique()]
        X_cell_test = mod.X_cell.loc[indices[test_idx]]
        X_cell_test = X_cell_test.loc[y_test_proc['Drug_CL'].unique()]
        
        y_train_proc['Drug_CL_Time'] = y_train_proc['Drug_CL'] + '_' + y_train_proc['Time'].astype(str)
        y_train_proc = y_train_proc.set_index('Drug_CL_Time').drop(columns=['Drug_CL','Time'])
        y_test_proc['Drug_CL_Time'] = y_test_proc['Drug_CL'] + '_' + y_test_proc['Time'].astype(str)
        y_test_proc = y_test_proc.set_index('Drug_CL_Time').drop(columns=['Drug_CL','Time'])
        
        # Shuffle data if specified
        if hyper_params['shuffle'] == True:
            orig_index = y_train_proc.index.copy()
            orig_columns = y_train_proc.columns.copy()

            # Shuffle the rows and cols
            row_order = np.random.permutation(y_train_proc.shape[0])
            shuffled_rows = y_train_proc.values[row_order, :]
            y_train_proc = pd.DataFrame(shuffled_rows, index=orig_index, columns=orig_columns)
            col_order = np.random.permutation(y_train_proc.shape[1])
            shuffled_cols = y_train_proc.values[:, col_order]
            y_train_proc = pd.DataFrame(shuffled_cols, index=y_train_proc.index, columns=orig_columns)
        
        mod_fold.y_out = y_train_proc
        
        # Train model
        start_time = time.time()
        model_trained, cur_loss, cur_eig, mean_loss, stats, X_train, X_test_, X_train_index, y_train, y_test_, y_train_index, X_cell_train, X_cell_test_, missing_node_indexes, floor_idx, ceil_idx, weight = train_signaling_model(
            mod_fold, net, optimizer, loss_fn, reset_epoch = 200, hyper_params = hyper_params, train_split_frac = {'train': 1, 'test': 0}, train_seed = seed, 
            verbose = True, split_by = 'condition', unique_time_points = [0, 1, 2, 3, 5, 8, 10, 50], noise_scale = 0.0)
        training_time = time.time() - start_time
        
        # Store training loss
        loss_smooth = utils.get_moving_average(values=stats['loss_mean'], n_steps=5)
        loss_sigma_smooth = utils.get_moving_average(values=stats['loss_sigma'], n_steps=10)
        epochs = np.arange(len(stats['loss_mean']))
        loss_df = pd.DataFrame({'loss_mean': loss_smooth, 'loss_sigma': loss_sigma_smooth}, index=epochs)
        
        # Benchmark training set
        y_actual = y_train
        Y_hat_train, Y_full_train, Y_fullFull_train, Y_fullprotein_train = model_trained(X_train, X_cell_train, missing_node_indexes)
        if hyper_params['use_time']:
            Y_subsampled = subsample_Y(Y_fullFull_train, floor_idx, ceil_idx, weight)
        else:
            unique_time_points = np.linspace(0, 149, 8).astype(int)
            Y_subsampled = Y_fullFull_train[:, unique_time_points, :]

        Y_subsampled = Y_subsampled.permute(1, 0, 2)
        y_actual = y_actual.reshape(8, len(X_train_index), mod.y_out.shape[1])

        y_train_index_CL = [entry.rsplit('_', 1)[0]
                            for entry in y_train_index
                            if entry.endswith('_0')]

        control_curves = identify_controls(
            Y_subsampled,    # your (T, N, P) predicted tensor on train
            y_actual,        # your (T, N, P) actual tensor on train
            y_train_index_CL,
            control_drug='D1'
        )

        Y_sub_train_adj, Y_act_train_adj = subtract_controls(
            Y_subsampled, y_actual, y_train_index_CL, control_curves
        )

        Y_sub_train_adj = Y_sub_train_adj - Y_sub_train_adj[0:1, :, :]
        Y_act_train_adj = Y_act_train_adj - Y_act_train_adj[0:1, :, :]

        y_pred_np = Y_sub_train_adj.detach().flatten().cpu().numpy()
        y_actual_np = Y_act_train_adj.detach().flatten().cpu().numpy()

        # Mask NaNs
        mask = ~np.isnan(y_pred_np) & ~np.isnan(y_actual_np)
        y_pred_filtered = y_pred_np[mask]
        y_actual_filtered = y_actual_np[mask]

        pr_train, _ = pearsonr(y_pred_filtered, y_actual_filtered)
        
        
        # Benchmark test set
        X_test = mod.df_to_tensor(X_test)
        X_cell_test = mod.df_to_tensor(X_cell_test)
        y_index = y_data.index.tolist()
        y_test_index = list(set(y_index) - set(y_train_index))
        y_actual_test = mod.df_to_tensor(y_test_proc)

        Y_hat_test, Y_full_test, Y_fullFull_test, Y_fullprotein_test = model_trained(X_test, X_cell_test, missing_node_indexes)
        if hyper_params['use_time']:
            Y_sub_test_ = subsample_Y(Y_fullFull_test, floor_idx, ceil_idx, weight)
        else:
            Y_sub_test_ = Y_fullFull_test[:, unique_time_points, :]

        Y_sub_test_ = Y_sub_test_.permute(1, 0, 2)
        y_actual_test = y_actual_test.reshape(8, len(X_test), mod.y_out.shape[1])

        y_test_index_CL = [entry.rsplit('_', 1)[0]
                            for entry in y_test_index
                            if entry.endswith('_0')]

        Y_sub_test_adj, Y_act_test_adj = subtract_controls(
            Y_sub_test_, y_actual_test, y_test_index_CL, control_curves
        )

        Y_sub_test_adj = Y_sub_test_adj - Y_sub_test_adj[0:1, :, :]
        Y_act_test_adj = Y_act_test_adj - Y_act_test_adj[0:1, :, :]

        y_pred_np = Y_sub_test_adj.detach().flatten().cpu().numpy()
        y_actual_np = Y_act_test_adj.detach().flatten().cpu().numpy()

        # Mask NaNs
        mask = ~np.isnan(y_pred_np) & ~np.isnan(y_actual_np)
        y_pred_filtered = y_pred_np[mask]
        y_actual_filtered = y_actual_np[mask]

        pr_test, _ = pearsonr(y_pred_filtered, y_actual_filtered)

        cv_results.append({
            'Config': config,
            'CV_set': fold + 1,
            'Train/Test': 'Train',
            'Performance': pr_train,
            'Time': training_time
        })
        cv_results.append({
            'Config': config,
            'CV_set': fold + 1,
            'Train/Test': 'Test',
            'Performance': pr_test,
            'Time': training_time
        })
        
    return cv_results


def train_full(mod, net, hyper_params, x_val, x_cell_val, y_val, config = None, cv_results = []):
    """
    Train the model on the full training set (mod.y_out) and evaluate on the test set.

    Parameters
    ----------
    mod : SignalingModel
        The model to be trained (initialized with the full training data).
    net : pd.DataFrame
        The prior knowledge signaling network.
    hyper_params : dict
        Hyperparameters for training.
    x_val : pd.DataFrame
        Test set input data.
    x_cell_val : pd.DataFrame
        Test set cell data.
    y_val : pd.DataFrame
        Test set output data.
    config : str
        The configuration name for the current run.
    cv_results : list
        A list to store the results of each fold.

    Returns
    -------
    results : dict
        Dictionary containing training and test evaluation results.
    """
    print("Training on the full dataset...")

    # Reset certain parts of the model (e.g., fix input layer scaling, spectral prescaling)
    mod.input_layer.weights.requires_grad = False
    mod.signaling_network.prescale_weights(target_radius=target_spectral_radius)

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam

    # Make a deepcopy for training so that original mod is preserved if needed
    mod_full = copy.deepcopy(mod)

    # Process training outputs (assumes mod_full.y_out has a column 'Drug_CL_Time')
    y_out_proc = mod_full.y_out.reset_index()
    y_out_proc['Time'] = y_out_proc['Drug_CL_Time'].str.split('_').str[-1].astype(int)
    y_out_proc['Drug_CL'] = y_out_proc['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
    y_train_proc = y_out_proc.drop(columns=['Drug_CL_Time']).sort_values(by=['Time', 'Drug_CL'])
    y_train_proc['Drug_CL_Time'] = y_train_proc['Drug_CL'] + '_' + y_train_proc['Time'].astype(str)
    y_train_proc = y_train_proc.set_index('Drug_CL_Time').drop(columns=['Drug_CL', 'Time'])
    mod_full.y_out = y_train_proc

    # Train model on the full training set
    start_time = time.time()
    model_trained, cur_loss, cur_eig, mean_loss, stats, X_train, X_test_, X_train_index, y_train, y_test_, y_train_index, X_cell_train, X_cell_test_, missing_node_indexes, floor_idx, ceil_idx, weight = train_signaling_model(
        mod, net, optimizer, loss_fn, reset_epoch = 200, hyper_params = hyper_params, train_split_frac = {'train': 1, 'test': 0}, train_seed = seed, 
        verbose = True, split_by = 'condition', unique_time_points = [0, 1, 2, 3, 5, 8, 10, 50], noise_scale = 0.0)
    training_time = time.time() - start_time

    
    # Benchmark training set
    y_actual = y_train
    Y_hat_train, Y_full_train, Y_fullFull_train, Y_fullprotein_train = model_trained(X_train, X_cell_train, missing_node_indexes)
    if hyper_params['use_time']:
        Y_subsampled = subsample_Y(Y_fullFull_train, floor_idx, ceil_idx, weight)
    else:
        unique_time_points = np.linspace(0, 149, 8).astype(int)
        Y_subsampled = Y_fullFull_train[:, unique_time_points, :]

    Y_subsampled = Y_subsampled.permute(1, 0, 2)
    y_actual = y_actual.reshape(8, len(X_train_index), mod.y_out.shape[1])

    y_train_index_CL = [entry.rsplit('_', 1)[0]
                        for entry in y_train_index
                        if entry.endswith('_0')]

    control_curves = identify_controls(
        Y_subsampled,    # your (T, N, P) predicted tensor on train
        y_actual,        # your (T, N, P) actual tensor on train
        y_train_index_CL,
        control_drug='D1'
    )

    Y_sub_train_adj, Y_act_train_adj = subtract_controls(
        Y_subsampled, y_actual, y_train_index_CL, control_curves
    )

    Y_sub_train_adj = Y_sub_train_adj - Y_sub_train_adj[0:1, :, :]
    Y_act_train_adj = Y_act_train_adj - Y_act_train_adj[0:1, :, :]

    y_pred_np = Y_sub_train_adj.detach().flatten().cpu().numpy()
    y_actual_np = Y_act_train_adj.detach().flatten().cpu().numpy()

    # Mask NaNs
    mask = ~np.isnan(y_pred_np) & ~np.isnan(y_actual_np)
    y_pred_filtered = y_pred_np[mask]
    y_actual_filtered = y_actual_np[mask]

    pr_train, _ = pearsonr(y_pred_filtered, y_actual_filtered)
    
    
    # Benchmark validation set
    x_val = x_val.reindex(columns=x_drug.columns, fill_value=0)
    y_index = y_val.index.tolist()
    y_actual_val = mod.df_to_tensor(y_data_val)
    X_val = mod.df_to_tensor(x_val)
    X_cell_val = mod.df_to_tensor(x_cell_val)

    Y_hat_val, Y_full_val, Y_fullFull_val, Y_fullprotein_val = model_trained(X_val, X_cell_val, missing_node_indexes)
    if hyper_params['use_time']:
        Y_subsampled_val = subsample_Y(Y_fullFull_val, floor_idx, ceil_idx, weight)
    else:
        unique_time_points = np.linspace(0, 149, 8).astype(int)
        Y_subsampled_val = Y_fullFull_val[:, unique_time_points, :]

    Y_subsampled_val = Y_subsampled_val.permute(1, 0, 2)
    y_actual_val = y_actual_val.reshape(8, len(X_val), mod.y_out.shape[1])

    y_val_index_CL = [entry.rsplit('_', 1)[0]
                        for entry in y_index
                        if entry.endswith('_0')]

    Y_sub_val_adj, Y_act_val_adj = subtract_controls(
        Y_subsampled_val, y_actual_val, y_val_index_CL, control_curves
    )

    Y_sub_val_adj = Y_sub_val_adj - Y_sub_val_adj[0:1, :, :]
    Y_act_val_adj = Y_act_val_adj - Y_act_val_adj[0:1, :, :]

    y_pred_np = Y_sub_val_adj.detach().flatten().cpu().numpy()
    y_actual_np = Y_act_val_adj.detach().flatten().cpu().numpy()

    # Mask NaNs
    mask = ~np.isnan(y_pred_np) & ~np.isnan(y_actual_np)
    y_pred_filtered = y_pred_np[mask]
    y_actual_filtered = y_actual_np[mask]

    pr_val, _ = pearsonr(y_pred_filtered, y_actual_filtered)
    
    cv_results.append({
            'Config': config,
            'CV_set': 'validation',
            'Train/Test': 'validation',
            'Performance': pr_val,
            'Time': training_time
        })
    
    return cv_results


n_cores = 12
utils.set_cores(n_cores)

seed = 49
if seed:
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    utils.set_seeds(seed = seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Prior knowledge signaling network
current_dir = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(current_dir, "data", "KEGGnet-Model.tsv")
net = pd.read_csv(data_path, sep = '\t', index_col = False)

# Synthetic data input and output
x_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_x_8.csv'), sep=',', low_memory=False, index_col=0)
x_cell = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xcell_8.csv'), sep=',', low_memory=False, index_col=0)
x_drug = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xdrug_comb_8.csv'), sep=',', low_memory=False, index_col=0)
y_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_y_scaled_8.csv'), sep=',', low_memory=False, index_col=0)
nodes_sites_map = pd.read_csv(os.path.join(current_dir, 'data', 'nodes_sites_map.csv'), sep=',', low_memory=False, index_col=0)

x_data_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_x_test.csv'), sep=',', low_memory=False, index_col=0)
x_cell_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xcell_test.csv'), sep=',', low_memory=False, index_col=0)
y_data_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_y_test.csv'), sep=',', low_memory=False, index_col=0)

G = nx.from_pandas_edgelist(net, source='source', target='target', create_using=nx.DiGraph())

start_nodes = ['P01133']

# Compute the reachable nodes from all start nodes.
reachable_nodes = set()
for node in start_nodes:
    if node in G:  # Only consider nodes present in the graph
        reachable_nodes.update(nx.descendants(G, node))
        reachable_nodes.add(node)
    else:
        print(f"Warning: Starting node {node} not found in graph.")

reachable_list = sorted(reachable_nodes)

# Filter dataframes
net = net[net['source'].isin(reachable_list) & net['target'].isin(reachable_list)]
proteins_net = pd.concat([net['source'], net['target']], ignore_index=True).unique().tolist()

nodes_sites_map = nodes_sites_map.loc[:, nodes_sites_map.columns.intersection(proteins_net)]
nodes_sites_map = nodes_sites_map[nodes_sites_map.sum(axis=1) != 0]

# Find common phosphosites between y_data and nodes_sites_map
common_phosphosites = pd.Index(y_data.columns).intersection(nodes_sites_map.index)

y_data = y_data.loc[:, common_phosphosites]
y_data_val = y_data_val.loc[:, common_phosphosites]
nodes_sites_map = nodes_sites_map.loc[common_phosphosites, :]

stimulation_label = 'stimulation'
inhibition_label = 'inhibition'
weight_label = 'mode_of_action'
source_label = 'source'
target_label = 'target'

net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

# linear scaling of inputs/outputs
projection_amplitude_in = 3
projection_amplitude_out = 1.2
# other parameters
bionet_params = {'target_steps': 100, 'max_steps': 150, 'exp_factor':50, 'tolerance': 1e-5, 'leak':1e-2} # fed directly to model

# training parameters
lr_params = {'max_iter': 3000, 
             'learning_rate': 2e-3,
             'variable_lr': True}
other_params = {'lambda_dynamic': 1, 'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-2}  # 1e-7 if not using variable gradient noise
regularization_params = {'param_lambda_L2': 1e-5, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5, 'lambda_simplify': 0}
spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
target_spectral_radius = 0.8
module_params = {
    'use_cln': True,
    'cln_hidden_layers': {1: 64, 2: 16},  # {1: 64, 2: 16}
    'use_xssn': True,
    'xssn_hidden_layers': {1: 16},
    'use_time': True,
    'n_timepoints': 8,
    'use_phospho': True,
    'nsl_hidden_layers': {1: 16, 2:8}, #{1: 16},
    'conn_dim': 5 #5
}

# Ablation study configurations
ablation_configs = [
    {'name': 'initial',  'use_time': False, 'use_phospho': False, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'time',  'use_time': True, 'use_phospho': False, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'phospho',  'use_time': True, 'use_phospho': True, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'bcell',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'xcell',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'hiddenlayers',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': {1: 64, 2: 16}, 'xssn_hidden_layers': {1: 16}},
    {'name': 'embedding',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': {1: 64, 2: 16}, 'xssn_hidden_layers': {1: 16}, 'conn_dim': 5, 'nsl_hidden_layers': {1: 16, 2:8}},
    {'name': 'random',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': {1: 64, 2: 16}, 'xssn_hidden_layers': {1: 64, 2: 16}, 'conn_dim': 5, 'nsl_hidden_layers': {1: 16, 2:8}, 'shuffle': True}
]

cv_results = []
for config in ablation_configs:
    print(f"Running configuration: {config['name']}")
    
    # Update module parameters for this ablation
    module_params['use_phospho'] = config['use_phospho']
    module_params['use_time'] = config['use_time']
    module_params['use_cln'] = config['use_cln']
    module_params['use_xssn'] = config['use_xssn']
    module_params['cln_hidden_layers'] = config['cln_hidden_layers']
    module_params['xssn_hidden_layers'] = config['xssn_hidden_layers']
    
    if 'conn_dim' in config:
        module_params['conn_dim'] = config['conn_dim']
        module_params['nsl_hidden_layers'] = config['nsl_hidden_layers']
    
    if 'shuffle' in config:
        module_params['shuffle'] = config['shuffle']
    
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params, **module_params}
    
    
    mod = SignalingModel(net = net,
                        X_in = x_data,
                        y_out = y_data, 
                        X_cell = x_cell,
                        X_drug = x_drug,
                        nodes_sites_map = nodes_sites_map,
                        projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                        weight_label = weight_label, source_label = source_label, target_label = target_label,
                        bionet_params = bionet_params, 
                        dtype = torch.float32, device = device, seed = seed, module_params = module_params)


    # Deactivate time layer if not used
    if not hyper_params.get('use_time', True):
        class NoOpModule(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, *args, **kwargs):
                return None
            def parameters(self):
                return []
        mod.time_layer = NoOpModule()

    # Train model with 5-fold cross-validation
    cv_results = train_cv(mod, net, hyper_params, n_cv=5, config=config['name'], cv_results=cv_results)
    cv_results = train_full(mod, net, hyper_params, x_data_val, x_cell_val, y_data_val, config=config['name'], cv_results=cv_results)

cv_results = pd.DataFrame(cv_results)
print(cv_results)

# Plot the results summary
config_labels = {
    "embedding": "Phospho-Embedding",
    "hiddenlayers": "Cell HLs",
    "xcell": "Cell Initialization",
    "bcell": "Cell Bias",
    "phospho": "Phosphosite Layer",
    "time": "Time Layer",
    "initial": "Baseline",
    "random": "Random"
}
config_order = list(config_labels.keys())

# Filter points and lines
points_df = cv_results[(cv_results['Train/Test'] != 'Train') & (cv_results['CV_set'] != 'validation')].copy()
lines_df = cv_results[(cv_results['CV_set'] == 'validation') & (cv_results['Train/Test'] != 'Train')].copy()

# Make sure config is categorical with proper order
cv_results['Config'] = pd.Categorical(cv_results['Config'], categories=config_order, ordered=True)
points_df['Config'] = pd.Categorical(points_df['Config'], categories=config_order, ordered=True)
lines_df['Config'] = pd.Categorical(lines_df['Config'], categories=config_order, ordered=True)

annot_df = (
    cv_results[
        (cv_results["CV_set"] == "validation") &
        (cv_results["Train/Test"] != "Train")
    ]
    .groupby("Config")
    .agg(
        xpos=("Performance", "mean"),
        time_min=("Time", lambda x: round(x.iloc[0] / 60, 1))
    )
    .reset_index()
)

annot_df["label"] = "Train time: " + annot_df["time_min"].astype(str) + " min"
annot_df["config"] = annot_df["Config"]
annot_df["xpos"] = -0.3

# Plot
p = (
    p9.ggplot()
    + p9.geom_jitter(
        data=points_df,
        mapping=p9.aes(x="Performance", y="Config", color="Config"),
        width=0, height=0.2, size=2.5
    )
    + p9.geom_line(
        data=lines_df,
        mapping=p9.aes(x="Performance", y="Config", group="Config", color="Config"),
        size=1
    )
    + p9.geom_text(
        data=annot_df,
        mapping=p9.aes(x="xpos", y="config", label="label"),
        size=10,
        va="bottom"
    )
    + p9.labs(x="Pearson Correlation", y="")
    + p9.scale_x_continuous(limits=(-0.5, 1), breaks=np.arange(0, 1.01, 0.2))
    + p9.scale_y_discrete(labels=config_labels)
    + p9.theme_bw()
    + p9.theme(
        figure_size=(10, 6),
        axis_text_y=p9.element_text(size=12),
        axis_text_x=p9.element_text(size=12),
        legend_position="none"
    )
)

p.show()

# Save results
os.makedirs(os.path.join(current_dir, 'results', 'ablation_test'), exist_ok=True)
pd.DataFrame(cv_results).to_csv(os.path.join(current_dir, 'results','ablation_test','ablation_test.csv'), index=False)

# Save plot
p.save(os.path.join(current_dir, 'results','ablation_test','ablation_test.pdf'), width=10, height=6, dpi=300)
