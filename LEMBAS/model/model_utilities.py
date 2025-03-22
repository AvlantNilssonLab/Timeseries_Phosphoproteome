"""
Helper functions for building the model.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

def np_to_torch(arr: np.array, dtype: torch.float32, device: str = 'cpu'):
    """Convert a numpy array to a torch.tensor

    Parameters
    ----------
    arr : np.array
        
    dtype : torch.dtype, optional
        datatype to store values in torch, by default torch.float32
    device : str
        whether to use gpu ("cuda") or cpu ("cpu"), by default "cpu"
    """
    return torch.tensor(arr, dtype=dtype, device = device)

def format_network(net: pd.DataFrame, 
                   weight_label: str = 'mode_of_action', 
                   stimulation_label: str = 'stimulation', 
                   inhibition_label: str = 'inhibition') -> pd.DataFrame:
    """Formats the standard network file format to that needed by `SignalingModel.parse_network`

    Parameters
    ----------
    net : pd.DataFrame
        signaling network adjacency list with the following columns:
            - `weight_label`: whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0.1). Exclude non-interacting (0) nodes. 
            - `stimulation_label`: binary whether an interaction is stimulating (1) or [not stimultaing or unknown] (0)
            - `inhibition_label`: binary whether an interaction is inhibiting (1) or [not inhibiting or unknown] (0)
    weight_label : str, optional
        converts `stimulation_label` and `inhibition_label` to a single column of stimulating (1), inhibiting (-1), or
        unknown (0.1), by default 'mode_of_action'
    stimulation_label : str, optional
        column name of stimulating interactions, see `net`, by default 'stimulation'
    inhibition_label : str, optional
        column name of inhibitory interactions, see `net`, by default 'inhibition'

    Returns
    -------
    formatted_net : pd.DataFrame
        the same dataframe with the additional `weight_label` column
    """
    if net[(net[stimulation_label] == 1) & (net[inhibition_label] == 1)].shape[0] > 0:
        raise ValueError('An interaction can either be stimulating (1,0), inhibition (0,1) or unknown (0,0)')
    
    formatted_net = net.copy()
    formatted_net[weight_label] = np.zeros(net.shape[0])
    formatted_net.loc[formatted_net[stimulation_label] == 1, weight_label] = 1
    formatted_net.loc[formatted_net[inhibition_label] == 1, weight_label] = -1
    
    #ensuring that lack of known MOA does not imply lack of representation in scipy.sparse.find(A)
    formatted_net[weight_label] = formatted_net[weight_label].replace(0, 0.1)
    formatted_net[weight_label] = formatted_net[weight_label].replace(np.nan, 0.1)

    return formatted_net

# def get_spectral_radius(weights: nn.parameter.Parameter):
#     """_summary_

#     Parameters
#     ----------
#     weights : nn.parameter.Parameter
#         the interaction weights

#     Returns
#     -------
#     spectral_radius : np.ndarray
#         a single element numpy array representing the denominator of the scaling factor for weights 
#     """
#     A = scipy.sparse.csr_matrix(weights.detach().numpy())
#     eigen_value, _ = eigs(A, k = 1) # first eigen value
#     spectral_radius = np.abs(eigen_value)
#     return spectral_radius

# def expected_uniform_distribution(Y_full: torch.Tensor, target_min: float = 0.0, target_max: float = None):
#     """Calculate the distance between the signaling network node values and a desired uniform distribution of the node values

#     Parameters
#     ----------
#     Y_full : torch.Tensor
#         the signaling network scaled by learned interaction weights. Shape is (samples x network nodes). 
#         Output of BioNet.
#     target_min : float, optional
#         minimum values for nodes in Y_full to take on, by default 0.0
#     target_max : float, optional
#         maximum values for nodes in Y_full to take on, by default 0.8

#     Returns
#     -------
#     loss : torch.Tensor
#         the regularization term
#     """
#     target_distribution = torch.linspace(target_min, target_max, Y_full.shape[0], dtype=Y_full.dtype, device=Y_full.device).reshape(-1, 1)

#     sorted_Y_full, _ = torch.sort(Y_full, axis=0) # sorts each column (signaling network node) in ascending order
#     dist_loss = torch.sum(torch.square(sorted_Y_full - target_distribution)) # difference in distribution
#     below_range = torch.sum(Y_full.lt(target_min) * torch.square(Y_full-target_min)) # those that are below the minimum value
#     above_range = torch.sum(Y_full.gt(target_max) * torch.square(Y_full-target_max)) # those that are above the maximum value
#     loss = dist_loss + below_range + above_range
#     return loss
