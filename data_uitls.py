import numpy as np
from sklearn.datasets import make_moons

import torch
from torch.utils.data import Dataset

def get_noisy_two_moons(num_sample, num_feature, noise_twomoon, noise_nuisance, seed):
    np.random.seed(seed)

    # X's shape = (1000, 2)
    # y's shape = (1000,)
    X, y = make_moons(n_samples=num_sample, noise=noise_twomoon, random_state=seed)

    # create additional irrelavent 8 features
    N = np.random.normal(loc=0., scale=noise_nuisance, size=[num_sample, num_feature-2])
    X = np.concatenate([X, N], axis=1)

    # one-hot encoding of y
    y_onehot = np.zeros([num_sample, 2])
    y_onehot[y == 0, 0] = 1
    y_onehot[y == 1, 1] = 1

    return X, y, y_onehot

def get_blockcorr(X, block_size, noise, seed):
    # add correlated 9 features to each feature in block
    for p in range(X.shape[1]):
        np.random.seed(seed + p)
        tmp = X[:, [p]] + np.random.normal(loc=0., scale=noise, size=[X.shape[0], block_size-1])
        if p == 0:
            X_new = np.concatenate([X[:, [p]], tmp], axis=1)
        else:
            X_new = np.concatenate([X_new, X[:, [p]], tmp], axis=1)    
    return X_new


class Phase_1_Dataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]