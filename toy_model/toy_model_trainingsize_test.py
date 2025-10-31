import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
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

# Load common validation set and node_so_sites_map
nodes_sites_map = pd.read_csv(os.path.join(current_dir, 'data', 'nodes_sites_map.csv'), sep=',', low_memory=False, index_col=0)
x_data_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_x_test.csv'), sep=',', low_memory=False, index_col=0)
x_cell_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_xcell_test.csv'), sep=',', low_memory=False, index_col=0)
y_data_val = pd.read_csv(os.path.join(current_dir, 'data', 'synthetic_data_y_test.csv'), sep=',', low_memory=False, index_col=0)

stimulation_label = 'stimulation'
inhibition_label = 'inhibition'
weight_label = 'mode_of_action'
source_label = 'source'
target_label = 'target'

net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)

# Filter nodes based on the networks input
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

hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params, **module_params}

train_overall = []
val_overall   = []
train_per_sample = []
val_per_sample   = []

for sz in [2, 8, 30, 50, 100]:
    suffix = f"synthetic_data_x_{sz}.csv"
    x_data   = pd.read_csv(os.path.join(current_dir,'data', suffix), index_col=0)
    x_cell   = pd.read_csv(os.path.join(current_dir,'data', f"synthetic_data_xcell_{sz}.csv"), index_col=0)
    x_drug   = pd.read_csv(os.path.join(current_dir,'data', f"synthetic_data_xdrug_comb_{sz}.csv"), index_col=0)
    y_data   = pd.read_csv(os.path.join(current_dir,'data', f"synthetic_data_y_scaled_{sz}.csv"), index_col=0)
    
    # Find common phosphosites between y_data and nodes_sites_map
    common_phosphosites = pd.Index(y_data.columns).intersection(nodes_sites_map.index)

    y_data = y_data.loc[:, common_phosphosites]
    y_data_val = y_data_val.loc[:, common_phosphosites]
    nodes_sites_map = nodes_sites_map.loc[common_phosphosites, :]

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

    # model setup
    mod.input_layer.weights.requires_grad = False # don't learn scaling factors for the ligand input concentrations
    mod.signaling_network.prescale_weights(target_radius = target_spectral_radius) # spectral radius

    # loss and optimizer
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam

    # training loop
    mod, cur_loss, cur_eig, mean_loss, stats, X_train, X_test, X_train_index, y_train, y_test, y_train_index, X_cell_train, X_cell_test, missing_node_indexes, floor_idx, ceil_idx, weight = train_signaling_model(
        mod, net, optimizer, loss_fn, reset_epoch = 200, hyper_params = hyper_params, train_split_frac = {'train': 1, 'test': 0}, train_seed = seed, 
        verbose = True, split_by = 'condition', unique_time_points = [0, 1, 2, 3, 5, 8, 10, 50], noise_scale = 0.0)


    # Benchmark training set
    y_actual = y_train
    Y_hat_train, Y_full_train, Y_fullFull_train, Y_fullprotein_train = mod(X_train, X_cell_train, missing_node_indexes)
    Y_subsampled = subsample_Y(Y_fullFull_train, floor_idx, ceil_idx, weight)

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
    train_overall.append({'Size6': sz*6, 'Dataset':'Train', 'Pearson_r': pr_train})

    train_results = []
    for i, sample_id in enumerate(y_train_index_CL):
        drug, cl = sample_id.split('_')
        
        y_pred_i = Y_sub_train_adj[:, i, :].detach().cpu().numpy().flatten()
        y_true_i = Y_act_train_adj[:, i, :].detach().cpu().numpy().flatten()
        
        # Mask NaNs
        mask = ~np.isnan(y_pred_i) & ~np.isnan(y_true_i)
        if np.sum(mask) == 0:
            continue
        
        r, _ = pearsonr(y_pred_i[mask], y_true_i[mask])
        train_results.append({
            'Pearson_r': r,
            'Drug': drug,
            'CellLine': cl
        })

    for rec in train_results:  # the list of {'Pearson_r','Drug','CellLine'}
        train_per_sample.append({**rec, 'Size6': sz*6, 'Dataset':'Train'})

    # Benchmark validation set
    x_data_val = x_data_val.reindex(columns=x_drug.columns, fill_value=0)
    y_index = y_data_val.index.tolist()
    y_actual_val = mod.df_to_tensor(y_data_val)
    X_val = mod.df_to_tensor(x_data_val)
    X_cell_val = mod.df_to_tensor(x_cell_val)

    Y_hat_val, Y_full_val, Y_fullFull_val, Y_fullprotein_val = mod(X_val, X_cell_val, missing_node_indexes)
    Y_subsampled_val = subsample_Y(Y_fullFull_val, floor_idx, ceil_idx, weight)

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
    val_overall.append({'Size6': sz*6, 'Dataset':'Val', 'Pearson_r': pr_val})

    val_results = []
    for i, sample_id in enumerate(y_val_index_CL):
        drug, cl = sample_id.split('_')
        
        y_pred_i = Y_sub_val_adj[:, i, :].detach().cpu().numpy().flatten()
        y_true_i = Y_act_val_adj[:, i, :].detach().cpu().numpy().flatten()
        
        # Mask NaNs
        mask = ~np.isnan(y_pred_i) & ~np.isnan(y_true_i)
        if np.sum(mask) == 0:
            continue
        
        r, _ = pearsonr(y_pred_i[mask], y_true_i[mask])
        val_results.append({
            'Pearson_r': r,
            'Drug': drug,
            'CellLine': cl
        })

    for rec in val_results:
            val_per_sample.append({**rec, 'Size6': sz*6, 'Dataset':'Val'})


# Save results
os.makedirs(os.path.join(current_dir, 'results', 'trainingsize_test'), exist_ok=True)
pd.DataFrame(train_overall).to_csv(os.path.join(current_dir, 'results','trainingsize_test','overall_train.csv'), index=False)
pd.DataFrame(val_overall).to_csv(os.path.join(current_dir, 'results','trainingsize_test','overall_val.csv'),   index=False)
pd.DataFrame(train_per_sample).to_csv(os.path.join(current_dir,'results','trainingsize_test','per_sample_train.csv'), index=False)
pd.DataFrame(val_per_sample).to_csv(os.path.join(current_dir,'results','trainingsize_test','per_sample_val.csv'),   index=False)

df_overall = pd.concat([pd.DataFrame(train_overall), pd.DataFrame(val_overall)], ignore_index=True)
df_samples = pd.concat([pd.DataFrame(train_per_sample), pd.DataFrame(val_per_sample)], ignore_index=True)

# Merge to associate overall Pearson r with each group
summary_df = (
    df_samples
    .groupby(['Size6', 'Dataset'])
    .agg(
        Std=('Pearson_r', 'std'),
        N=('Pearson_r', 'count')
    )
    .reset_index()
)

# Bring in the overall Pearson as the mean line
df_overall_renamed = df_overall.rename(columns={'Pearson_r': 'MeanPearson'})
summary_df = summary_df.merge(
    df_overall_renamed[['Size6', 'Dataset', 'MeanPearson']],
    on=['Size6', 'Dataset']
)

# 95% CI around the overall Pearson using StdErr
summary_df['CI_lower'] = summary_df['MeanPearson'] - 1.96 * summary_df['Std'] / np.sqrt(summary_df['N'])
summary_df['CI_upper'] = summary_df['MeanPearson'] + 1.96 * summary_df['Std'] / np.sqrt(summary_df['N'])


custom_palette = {
    'Train': '#4091C9',
    'Val':   '#CB1B16'
}

p = (
    p9.ggplot(summary_df, p9.aes(x='Size6', y='MeanPearson', color='Dataset', fill='Dataset'))
    + p9.geom_line(size=1.2)
    + p9.geom_ribbon(p9.aes(ymin='CI_lower', ymax='CI_upper'), alpha=0.2)
    + p9.scale_color_manual(values=custom_palette)
    + p9.scale_fill_manual(values=custom_palette)
    + p9.scale_x_continuous(limits=(0, 600), breaks=range(0, 601, 100))
    + p9.scale_y_continuous(limits=(0, 1), breaks=np.linspace(0, 1, 11))
    + p9.labs(x='Training Data Size', y='Test Correlation')
    + p9.theme_bw()
)

p.save(os.path.join(current_dir, 'results','trainingsize_test','trainingsize_test.pdf'), width=10, height=6, dpi=300)