import os
import pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import time

from scipy.stats import pearsonr
import torch
from sklearn.model_selection import KFold
import copy

import plotnine as p9

import sys

# Add the project root (one directory up from the current file) to sys.path
lem_bas_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "LEMBAS"))
if lem_bas_dir not in sys.path:
    sys.path.insert(0, lem_bas_dir)

# Now import using the package name
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
from LEMBAS import utilities as utils
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


def train_cv(mod, net, hyper_params, n_cv = 5):
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
    
    Returns
    -------
    cv_results : list
        The results of the cross-validation.
    """
    cv_results = []  # Store results for each fold

    # Prepare 5xFolds
    indices = np.array(mod.X_in.index)
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=888)

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"Processing fold {fold+1}...")
        
        # Reset model setup for each fold
        mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
        mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam

        mod_fold = copy.deepcopy(mod)
        mod_fold.X_in = mod.X_in.loc[indices[train_idx]]
        mod_fold.X_cell = mod.X_cell.loc[indices[train_idx]]
        
        #mod_fold.y_out = mod.y_out.loc[indices[train_idx]]
        y_out_proc = mod.y_out.reset_index()
        y_out_proc['Time'] = y_out_proc['Drug_CL_Time'].str.split('_').str[-1].astype(int)
        y_out_proc['Drug_CL'] = y_out_proc['Drug_CL_Time'].str.rsplit('_', n=1).str[0]
        y_train_proc = y_out_proc[y_out_proc['Drug_CL'].isin(indices[train_idx])]
        y_test_proc = y_out_proc[y_out_proc['Drug_CL'].isin(indices[test_idx])]
        y_train_proc = y_train_proc.drop(columns=['Drug_CL_Time']).sort_values(by=['Time', 'Drug_CL'])
        y_test_proc = y_test_proc.drop(columns=['Drug_CL_Time']).sort_values(by=['Time', 'Drug_CL'])
        y_train_proc['Drug_CL_Time'] = y_train_proc['Drug_CL'] + '_' + y_train_proc['Time'].astype(str)
        y_train_proc = y_train_proc.set_index('Drug_CL_Time').drop(columns=['Drug_CL','Time'])
        y_test_proc['Drug_CL_Time'] = y_test_proc['Drug_CL'] + '_' + y_test_proc['Time'].astype(str)
        y_test_proc = y_test_proc.set_index('Drug_CL_Time').drop(columns=['Drug_CL','Time'])
        mod_fold.y_out = y_train_proc
        
        # Train model
        start_time = time.time()
        model_trained, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_train_index, y_train, y_test, y_train_index, X_cell_train, X_cell_test, missing_node_indexes, floor_idx, ceil_idx, weight = train_signaling_model(
            mod_fold, net, optimizer, loss_fn, 
            reset_epoch=200,
            hyper_params=hyper_params,
            train_split_frac={'train': 1.0, 'test': 0.0},
            train_seed=888,
            split_by='condition', 
            noise_scale=0
        )
        training_time = time.time() - start_time
        
        # Store training loss
        loss_smooth = utils.get_moving_average(values=stats['loss_mean'], n_steps=5)
        loss_sigma_smooth = utils.get_moving_average(values=stats['loss_sigma'], n_steps=10)
        epochs = np.arange(len(stats['loss_mean']))
        loss_df = pd.DataFrame({'loss_mean': loss_smooth, 'loss_sigma': loss_sigma_smooth}, index=epochs)
        
        # Evaluation on training data
        Y_hat_train, Y_full_train, Y_fullFull_train = model_trained(X_train, X_cell_train, missing_node_indexes)
        if hyper_params['use_time']:
            Y_sub_train = subsample_Y(Y_fullFull_train, floor_idx, ceil_idx, weight)
        else:
            unique_time_points = np.linspace(0, 149, 8).astype(int)
            Y_sub_train = Y_fullFull_train[:, unique_time_points, :]
        
        Y_sub_train = Y_sub_train - Y_sub_train[:, 0:1, :]
        Y_sub_train = Y_sub_train.permute(1, 0, 2)
        Y_sub_train = torch.flatten(Y_sub_train, start_dim=0, end_dim=1)
        
        y_actual_train = y_train.reshape(8, len(X_train_index), mod.y_out.shape[1])
        y_actual_train = y_actual_train - y_actual_train[0:1, :, :]
        y_actual_train = torch.flatten(y_actual_train, start_dim=0, end_dim=1)
        
        y_pred_train_np = Y_sub_train.detach().cpu().numpy().flatten()
        y_actual_train_np = y_actual_train.detach().cpu().numpy().flatten()
        mask_train = ~np.isnan(y_pred_train_np) & ~np.isnan(y_actual_train_np)
        y_pred_train_filtered = y_pred_train_np[mask_train]
        y_actual_train_filtered = y_actual_train_np[mask_train]
        
        pr_train, _ = pearsonr(y_pred_train_filtered, y_actual_train_filtered)
        
        train_df = pd.DataFrame({'True': y_actual_train_filtered, 'Predicted': y_pred_train_filtered})
        
        # Evaluation on Test Data
        X_test = mod.X_in.loc[indices[test_idx]]
        X_cell_test = mod.X_cell.loc[indices[test_idx]]
        X_test = mod.df_to_tensor(X_test)
        X_cell_test = mod.df_to_tensor(X_cell_test)
        Y_hat_test, Y_full_test, Y_fullFull_test = model_trained(X_test, X_cell_test, missing_node_indexes)
        if hyper_params['use_time']:
            Y_sub_test = subsample_Y(Y_fullFull_test, floor_idx, ceil_idx, weight)
        else:
            Y_sub_test = Y_fullFull_test[:, unique_time_points, :]
        
        Y_sub_test = Y_sub_test - Y_sub_test[:, 0:1, :]
        Y_sub_test = Y_sub_test.permute(1, 0, 2)
        Y_sub_test = torch.flatten(Y_sub_test, start_dim=0, end_dim=1)
        
        y_test = mod.df_to_tensor(y_test_proc)
        y_actual_test = y_test.reshape(8, len(indices[test_idx]), mod.y_out.shape[1])
        y_actual_test = y_actual_test - y_actual_test[0:1, :, :]
        y_actual_test = torch.flatten(y_actual_test, start_dim=0, end_dim=1)
        
        y_pred_test_np = Y_sub_test.detach().cpu().numpy().flatten()
        y_actual_test_np = y_actual_test.detach().cpu().numpy().flatten()
        mask_test = ~np.isnan(y_pred_test_np) & ~np.isnan(y_actual_test_np)
        y_pred_test_filtered = y_pred_test_np[mask_test]
        y_actual_test_filtered = y_actual_test_np[mask_test]
        
        pr_test, _ = pearsonr(y_pred_test_filtered, y_actual_test_filtered)
        test_df = pd.DataFrame({'True': y_actual_test_filtered, 'Predicted': y_pred_test_filtered})
        
        # Save fold results
        cv_results.append({
            "fold": fold+1,
            "train": {"data": train_df, "pearson": pr_train, "loss": loss_df, "training_time": training_time},
            "test": {"data": test_df, "pearson": pr_test}
        })
        
        print(f"Fold {fold+1}: Train Pearson r: {pr_train:.2f}, Test Pearson r: {pr_test:.2f}")
    
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
x_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_x.csv'), sep=',', low_memory=False, index_col=0)
x_cell = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xcell.csv'), sep=',', low_memory=False, index_col=0)
x_drug = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xdrug.csv'), sep=',', low_memory=False, index_col=0)
y_data = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_y.csv'), sep=',', low_memory=False, index_col=0)
nodes_sites_map = pd.read_csv(os.path.join(current_dir, 'data', 'nodes_sites_map.csv'), sep=',', low_memory=False, index_col=0)


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
             'learning_rate': 2e-3}
other_params = {'batch_size': 10, 'noise_level': 10, 'gradient_noise_level': 1e-9}
regularization_params = {'param_lambda_L2': 1e-6, 'moa_lambda_L1': 0.1, 'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-4, 
                   'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': 1e-5}
spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
target_spectral_radius = 0.8
module_params = {
    'use_cln': True,
    'cln_hidden_layers': {1: 64, 2: 16},  # {1: 64, 2: 32}
    'use_xssn': True,
    'xssn_hidden_layers': {1: 64, 2: 16},
    'nsl_hidden_layers': None,
    'use_time': False,
    'n_timepoints': 8,
    'use_phospho': False
}

# Ablation study configurations
ablation_configs = [
    {'name': 'initial',  'use_time': False, 'use_phospho': False, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'with_time',  'use_time': True, 'use_phospho': False, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'with_time_with_phospho',  'use_time': True, 'use_phospho': True, 'use_cln': False, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'with_time_with_phospho_with_bcell',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': False, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'with_all',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': None, 'xssn_hidden_layers': None},
    {'name': 'with_all_hiddenlayers_in_cell',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': {1: 64, 2: 16}, 'xssn_hidden_layers': {1: 64, 2: 16}},
    {'name': 'random',  'use_time': True, 'use_phospho': True, 'use_cln': True, 'use_xssn': True, 'cln_hidden_layers': {1: 64, 2: 16}, 'xssn_hidden_layers': {1: 64, 2: 16}, 'shuffle': True}
]

for config in ablation_configs:
    print(f"Running configuration: {config['name']}")
    
    # Update module parameters for this ablation
    module_params['use_phospho'] = config['use_phospho']
    module_params['use_time'] = config['use_time']
    module_params['use_cln'] = config['use_cln']
    module_params['use_xssn'] = config['use_xssn']
    module_params['cln_hidden_layers'] = config['cln_hidden_layers']
    module_params['xssn_hidden_layers'] = config['xssn_hidden_layers']
    
    hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params, **module_params}

    # Shuffle data if specified
    if config['name'] == 'random':
        y_data = y_data.sample(frac=1, random_state=seed)
    
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
    cv_results = train_cv(mod, net, hyper_params, n_cv=5)
    
    print("Saving results...")
    config_results_path = os.path.join(current_dir, "results", "ablation_study_res", f"cv_results_{config['name']}.pkl")
    data_path = os.path.join(config_results_path)
    with open(data_path, "wb") as f:
        pickle.dump(cv_results, f)

'''# Load results
config_results_path = os.path.join(current_dir, "results", "ablation_study_res", "cv_results_with_time_with_phospho_with_bcell.pkl")
data_path = os.path.join(config_results_path)
with open(data_path, "rb") as f:
    cv_results_load = pickle.load(f)


# Visualize training loss of 1st fold
p1 = plotting.shade_plot(X = cv_results_load[0]["train"]["loss"].index, Y = cv_results_load[0]["train"]["loss"]['loss_mean'], sigma = cv_results_load[0]["train"]["loss"]['loss_sigma'], x_label = 'Epoch', y_label = 'Loss')
p1 += p9.scale_y_log10()
p1.show()

# Visualize training and test data correlation between predicted and true values for all folds
cv_dfs_train = []
for i in range(len(cv_results_load)):
    df_temp = cv_results_load[i]["test"]["data"].copy()
    df_temp["CV_set"] = i + 1
    cv_dfs_train.append(df_temp)
viz_df_train = pd.concat(cv_dfs_train, axis=0)
min_val = min(viz_df_train['Predicted'].min(), viz_df_train['True'].min())
max_val = max(viz_df_train['Predicted'].max(), viz_df_train['True'].max())
line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})

pearson_df_train = pd.DataFrame({
    "CV_set": list(range(1, len(cv_results_load)+1)),
    "label": [f"Pearson r: {cv_results_load[i]['train']['pearson']:.2f}" for i in range(len(cv_results_load))]
})
pearson_df_train["x"] = min_val + 0.25*(max_val-min_val)
pearson_df_train["y"] = max_val - 0.05*(max_val-min_val)

width, height = 8, 5
p2 = (
    p9.ggplot() +
    p9.geom_point(data = viz_df_train, mapping = p9.aes(x='Predicted', y = 'True'), color = '#1E90FF', alpha = 0.1) +
    p9.facet_wrap('~ CV_set') +
    p9.geom_line(data=line_df, mapping=p9.aes(x='x', y='y'), color='black') +
    p9.theme_bw() + 
    p9.theme(figure_size=(width, height)) +
    p9.geom_text(data=pearson_df_train, mapping=p9.aes(x='x', y='y', label='label'), size=10)
)
p2.show()

cv_dfs_test = []
for i in range(len(cv_results_load)):
    df_temp = cv_results_load[i]["test"]["data"].copy()
    df_temp["CV_set"] = i + 1
    cv_dfs_test.append(df_temp)
viz_df_test = pd.concat(cv_dfs_test, axis=0)
min_val = min(viz_df_test['Predicted'].min(), viz_df_test['True'].min())
max_val = max(viz_df_test['Predicted'].max(), viz_df_test['True'].max())
line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})

pearson_df_test = pd.DataFrame({
    "CV_set": list(range(1, len(cv_results_load)+1)),
    "label": [f"Pearson r: {cv_results_load[i]['test']['pearson']:.2f}" for i in range(len(cv_results_load))]
})
pearson_df_test["x"] = min_val + 0.25*(max_val-min_val)
pearson_df_test["y"] = max_val - 0.05*(max_val-min_val)

width, height = 8, 5
p3 = (
    p9.ggplot() +
    p9.geom_point(data = viz_df_test, mapping = p9.aes(x='Predicted', y = 'True'), color = '#1E90FF', alpha = 0.1) +
    p9.facet_wrap('~ CV_set') +
    p9.geom_line(data=line_df, mapping=p9.aes(x='x', y='y'), color='black') +
    p9.theme_bw() + 
    p9.theme(figure_size=(width, height)) +
    p9.geom_text(data=pearson_df_test, mapping=p9.aes(x='x', y='y', label='label'), size=10)
)
p3.show()'''