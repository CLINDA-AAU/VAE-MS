
import torch
from sklearn.decomposition import NMF
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset


def poisnll(output, target):
   # as gamma(k+1) = k! for integers
   zeros = (output == 0).sum()
   output[output == 0] =+ 1e-8
   loss = output - target*torch.log(output) + torch.lgamma(target + 1)

   return loss.mean()

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# He Initialization
def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def create_datasets(catalogue, train_idx, validation_idx, test_idx, lamdba_prior_train, lamdba_prior_val , lamdba_prior_test):
    train_data = torch.tensor(catalogue.values[train_idx, :], dtype=torch.float32)
    validation_data = torch.tensor(catalogue.values[validation_idx, :], dtype=torch.float32)
    test_data = torch.tensor(catalogue.values[test_idx, :], dtype=torch.float32)
    
    train_data = TensorDataset(
        train_data,
        torch.tensor(lamdba_prior_train.values, dtype=torch.float32))
    validation_data = TensorDataset(
        validation_data,
        torch.tensor(lamdba_prior_val.values, dtype=torch.float32))
    test_data = TensorDataset(
        test_data,
        torch.tensor(lamdba_prior_test.values, dtype=torch.float32)
    )
    return train_data, validation_data, test_data

# Helper function to initialize NMF priors
def initialize_nmf(catalogue, train_idx, validation_idx, test_idx, h_dim, tol = 1e-8):
    nmf = NMF(n_components=h_dim, init='random', solver='mu', beta_loss='kullback-leibler', max_iter=1000000, tol = tol)
    lamdba_prior_train = pd.DataFrame(nmf.fit_transform(catalogue.iloc[train_idx, :]))
    lamdba_prior_val = pd.DataFrame(nmf.transform(catalogue.iloc[validation_idx, :]))
    lamdba_prior_test = pd.DataFrame(nmf.transform(catalogue.iloc[test_idx, :]))
    
    start_sigs = pd.DataFrame(nmf.components_)
    diagonals = start_sigs.sum(axis=1)
    start_sigs = np.diag(1 / diagonals) @ start_sigs
    lamdba_prior_train = lamdba_prior_train @ np.diag(diagonals)
    lamdba_prior_val = lamdba_prior_val @ np.diag(diagonals)
    lamdba_prior_test = lamdba_prior_test @ np.diag(diagonals)

    return lamdba_prior_train, lamdba_prior_val, lamdba_prior_test, start_sigs, nmf.reconstruction_err_
