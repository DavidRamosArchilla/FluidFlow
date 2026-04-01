import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset


NWALLP = 260774  # number of points on the aircraft skin

def load_onera_crm(data_dir, pad_length=None):
    X_train_tot = np.load(data_dir + '/X_train.npy')
    Y_train_tot = np.load(data_dir + '/Ytrain.npy')
    X_test = np.load(data_dir + '/X_test.npy')
    Y_test = np.load(data_dir + '/Ytest.npy')

    df_description = pd.read_csv(data_dir + '/describe_train_test_repartition_with_weights.csv', index_col=0)
    df_test = df_description.loc[~df_description['Train']]
    df_train = df_description.loc[df_description['Train']]

    ncase = len(df_description)  # 468
    ntest = len(df_test) # 156
    ntrain = ncase-ntest  # 312

    # Remove the geometric informations from the input array
    X_train_tot_conditions = X_train_tot[0::NWALLP,6:9]
    X_test_conditions = X_test[0::NWALLP,6:9]
    # Create the output array to be of shape (ntrain, nwallp, 4)
    Y_train_tot_conditions = np.array([Y_train_tot[NWALLP*i:NWALLP*(i+1),:] for i in range(ntrain)])
    Y_test_tot_conditions = np.array([Y_test[NWALLP*i:NWALLP*(i+1),:] for i in range(ntest)])

    print("X_train_tot_conditions shape", X_train_tot_conditions.shape)
    print("Y_train_tot_conditions shape", Y_train_tot_conditions.shape)

    # split X_train_tot and Y_train_tot into train and validation arrays
    X_train = X_train_tot_conditions
    X_test = X_test_conditions

    Y_train = Y_train_tot_conditions
    Y_test = Y_test_tot_conditions

    Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 2, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1)

    # normalize/standarize things
    train_mean = Y_train.mean(dim=(0, 2), keepdim=True)
    train_std  = Y_train.std(dim=(0, 2), keepdim=True)
    condition_mean = X_train.mean(axis=0, keepdims=True)
    condition_std = X_train.std(axis=0, keepdims=True) 

    Y_train = (Y_train - train_mean) / train_std
    Y_test = (Y_test - train_mean) / train_std
    X_train = (X_train - condition_mean) / condition_std
    X_test = (X_test - condition_mean) / condition_std

    # pad sequences to a multple of a power of 2. 260864 = 256 * 1019
    if pad_length is not None:
        Y_train = F.pad(Y_train, (0, pad_length - NWALLP))
        Y_test = F.pad(Y_test, (0, pad_length - NWALLP))

    print("X train shape", X_train.shape)
    print("X test shape", X_test.shape)
    print("Y train shape", Y_train.shape)
    print("Y test shape", Y_test.shape)
    print("mean/std X train test ", X_train.mean(axis=0), X_test.mean(axis=0), X_train.std(axis=0), X_test.std(axis=0))
    print("mean/std Y train test ", Y_train.mean(dim=(0, 2)), Y_test.mean(dim=(0, 2)), Y_train.std(dim=(0, 2)), Y_test.std(dim=(0, 2)))
    dataset_train = TensorDataset(
        Y_train, torch.tensor(X_train, dtype=torch.float32)
    )

    dataset_test = TensorDataset(
        Y_test, torch.tensor(X_test, dtype=torch.float32)
    )
    coefficients = {
        'train_mean': train_mean,
        'train_std': train_std,
        'condition_mean': condition_mean,
        'condition_std': condition_std,
    }
    return dataset_train, dataset_test, coefficients