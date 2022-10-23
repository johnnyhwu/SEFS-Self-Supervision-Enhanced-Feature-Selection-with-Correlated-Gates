# %%
# import package
import random
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_uitls
import model_utils
import eval_utils

# %%
SEED = 1234
random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# create twomoon dataset for phase 1 and phase 2

# dataset with 1000 samples and a block of features (10 features in total)
# only #1 and #2 features are real, the other 8 features are sampled from gaussain distribution
X_train_phase_1, y_train_phase_1, y_onehot_train_phase_1 = data_uitls.get_noisy_two_moons(num_sample=2000, num_feature=10, noise_twomoon=0.1, noise_nuisance=1.0, seed=SEED)
X_train_phase_2, y_train_phase_2, y_onehot_train_phase_2 = data_uitls.get_noisy_two_moons(num_sample=1000, num_feature=10, noise_twomoon=0.1, noise_nuisance=1.0, seed=SEED+1)
X_valid_phase_2, y_valid_phase_2, y_onehot_valid_phase_2 = data_uitls.get_noisy_two_moons(num_sample=1000, num_feature=10, noise_twomoon=0.1, noise_nuisance=1.0, seed=SEED+2)
X_test_phase_2, y_test_phase_2, y_onehot_test_phase_2 = data_uitls.get_noisy_two_moons(num_sample=1000, num_feature=10, noise_twomoon=0.1, noise_nuisance=1.0, seed=SEED+3)

# add additional 9 features to each feature in block
# therefore, we have dataset with 1000 samples and 10 blocks of features (100 features in total)
# only #1 feature in #1 and #2 block are strongly correlated to target
# only #1 and #2 block of features are weakly correlated to target   
X_train_phase_1 = data_uitls.get_blockcorr(X=X_train_phase_1, block_size=10, noise=0.3, seed=SEED)
X_train_phase_2 = data_uitls.get_blockcorr(X=X_train_phase_2, block_size=10, noise=0.3, seed=SEED+1)
X_valid_phase_2 = data_uitls.get_blockcorr(X=X_valid_phase_2, block_size=10, noise=0.3, seed=SEED+2)
X_test_phase_2 = data_uitls.get_blockcorr(X=X_test_phase_2, block_size=10, noise=0.3, seed=SEED+3)

# only contain small number (20) of (labeled) smaples in X_train_phase_2
idx_1 = random.sample(np.where(y_train_phase_2 == 1)[0].tolist(), 10)
idx_0 = random.sample(np.where(y_train_phase_2 == 0)[0].tolist(), 10)
idx  = idx_1 + idx_0
random.shuffle(idx)

X_train_phase_2 = X_train_phase_2[idx]
y_train_phase_2 = y_train_phase_2[idx]
y_onehot_train_phase_2 = y_onehot_train_phase_2[idx]

# %%
# rescale features
scaler = MinMaxScaler()
scaler.fit(np.concatenate([X_train_phase_1, X_train_phase_2], axis=0))

X_train_phase_1 = scaler.transform(X_train_phase_1) 
X_train_phase_2 = scaler.transform(X_train_phase_2)
X_valid_phase_2 = scaler.transform(X_valid_phase_2)
X_test_phase_2 = scaler.transform(X_test_phase_2)

# %%
print("Shape of dataset:")
print("----------")
print(f"X_train (pahse-1): {X_train_phase_1.shape}")
print(f"y_onehot_train (pahse-1): {y_onehot_train_phase_1.shape}")
print("----------")
print(f"X_train (pahse-2): {X_train_phase_2.shape}")
print(f"y_onehot_train (pahse-2): {y_onehot_train_phase_2.shape}")
print("----------")
print(f"X_valid (pahse-2): {X_valid_phase_2.shape}")
print(f"y_onehot_valid (pahse-2): {y_onehot_valid_phase_2.shape}")
print("----------")
print(f"X_test (pahse-2): {X_test_phase_2.shape}")
print(f"y_onehot_test (pahse-2): {y_onehot_test_phase_2.shape}")

#%%
# cholesky decomposition of covariance matrix
cov = np.corrcoef(X_train_phase_1.T)
COV_L = scipy.linalg.cholesky(cov, lower=True)

def mask_generation(batch_size, pi):
    num_features = COV_L.shape[0]
    epsilon = np.random.normal(loc=0., scale=1., size=[num_features, batch_size])
    g = np.matmul(COV_L, epsilon)

    # CDF of standard normal distribution : Phi(x; 0,1) = 1/2 * (1 + erf( x/sqrt(2))) 
    g_normalized = 1/2 * (1 + scipy.special.erf(g / np.sqrt(2)))
    m = (g_normalized < pi).astype(float).T 
    return m

def copula_generation(batch_size):
    num_features = COV_L.shape[0]
    epsilon = np.random.normal(loc=0., scale=1., size=[num_features, batch_size])
    g = np.matmul(COV_L, epsilon)
    return g.T

#%%
# phase 1: self-supervised learning
model_phase_1 = model_utils.Phase_1_Model()
model_phase_1.to(DEVICE)
model_phase_1.train()

train_dataset = data_uitls.Phase_1_Dataset(X=X_train_phase_1[1000:])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = data_uitls.Phase_1_Dataset(X=X_train_phase_1[:1000])
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model_phase_1.parameters(), lr=1e-4)

#%%
input_mean = np.mean(X_train_phase_1, axis=0, keepdims=True)
fprob_threshold = 0.5
alpha = 10.0
model_update_count = 0
min_valid_loss = float("inf")
valid_patience = 0

#%%
while model_update_count < 500000 and valid_patience < 7:

    train_step_count = 0
    train_loss_recon_x = 0
    train_loss_recon_m = 0
    train_loss = 0

    for input_batch, label_batch in train_dataloader:
        
        # prepare input to model
        input_mean_batch = torch.from_numpy(np.tile(input_mean, [input_batch.shape[0], 1]))
        mask_batch = torch.from_numpy(mask_generation(input_batch.shape[0], fprob_threshold))

        # move input to DEVICE
        input_batch = input_batch.to(torch.float32).to(DEVICE)
        input_mean_batch = input_mean_batch.to(torch.float32).to(DEVICE)
        mask_batch = mask_batch.to(torch.float32).to(DEVICE)
        label_batch = label_batch.to(torch.float32).to(DEVICE)

        # zero out gradient
        optimizer.zero_grad()

        # calculate loss
        output_1_batch, output_2_batch = model_phase_1(input_batch, input_mean_batch, mask_batch)
        loss_recon_x = torch.mean(torch.sum((output_1_batch - label_batch)**2, axis=1))
        loss_recon_m = torch.mean(-torch.sum(mask_batch * eval_utils.log(output_2_batch) + (1. - mask_batch) * eval_utils.log(1. - output_2_batch), axis=1))            
        loss_main = loss_recon_x + alpha * loss_recon_m

        # update parameters in model
        loss_main.backward()
        optimizer.step()

        # save loss info
        train_loss_recon_x += loss_recon_x
        train_loss_recon_m += loss_recon_m
        train_loss += loss_main
        train_step_count += 1
        model_update_count += 1

        # validate model every 1000 iterations
        if model_update_count % 1000 == 0:

            # display training info
            print(f"step: {model_update_count:06} | tr_recon_x: {train_loss_recon_x / train_step_count:.2f} tr_recon_m: {train_loss_recon_m / train_step_count:.2f} tr_loss: {train_loss / train_step_count:.2f}")
            train_loss_recon_x = 0
            train_loss_recon_m = 0
            train_loss = 0
            train_step_count = 0

            # model validation
            model_phase_1.eval()
            
            valid_loss_recon_x = 0
            valid_loss_recon_m = 0
            valid_loss = 0
            
            for input_batch, label_batch in valid_dataloader:

                # prepare input to model
                input_mean_batch = torch.from_numpy(np.tile(input_mean, [input_batch.shape[0], 1]))
                mask_batch = torch.from_numpy(mask_generation(input_batch.shape[0], fprob_threshold))

                # move input to DEVICE
                input_batch = input_batch.to(torch.float32).to(DEVICE)
                input_mean_batch = input_mean_batch.to(torch.float32).to(DEVICE)
                mask_batch = mask_batch.to(torch.float32).to(DEVICE)
                label_batch = label_batch.to(torch.float32).to(DEVICE)

                # calculate loss
                output_1_batch, output_2_batch = model_phase_1(input_batch, input_mean_batch, mask_batch)
                loss_recon_x = torch.mean(torch.sum((output_1_batch - label_batch)**2, axis=1))
                loss_recon_m = torch.mean(-torch.sum(mask_batch * eval_utils.log(output_2_batch) + (1. - mask_batch) * eval_utils.log(1. - output_2_batch), axis=1))            
                loss_main = loss_recon_x + alpha * loss_recon_m

                # save loss info
                valid_loss_recon_x += loss_recon_x
                valid_loss_recon_m += loss_recon_m
                valid_loss += loss_main
            
            valid_loss_recon_x /= len(valid_dataloader)
            valid_loss_recon_m /= len(valid_dataloader)
            valid_loss /= len(valid_dataloader)

            print(f"step: {model_update_count:06} | va_recon_x: {valid_loss_recon_x:.2f} va_recon_m: {valid_loss_recon_m:.2f} va_loss: {valid_loss:.2f}")

            if valid_loss < min_valid_loss:
                print(f"Save encoder !")
                torch.save(model_phase_1.enc.state_dict(), "checkpoint/encoder.pt")
                min_valid_loss = valid_loss
                valid_patience = 0
            else:
                valid_patience += 1
            
            if valid_patience >= 7:
                break

            model_phase_1.train()

#%%
# phase 2: supervised learning
model_phase_2 = model_utils.Phase_2_Model(enc_path="checkpoint/encoder.pt")
model_phase_2.to(DEVICE)
model_phase_2.train()

train_dataset = data_uitls.Phase_2_Dataset(X=X_train_phase_2, y=y_onehot_train_phase_2)
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

valid_dataset = data_uitls.Phase_2_Dataset(X=X_valid_phase_2, y=y_onehot_valid_phase_2)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model_phase_2.parameters(), lr=1e-4)

#%%
beta = 1.0
model_update_count = 0
min_valid_loss = float("inf")
valid_patience = 0

#%%
while model_update_count < 500000 and valid_patience < 7:

    train_step_count = 0
    train_loss = 0

    for input_batch, label_batch in train_dataloader:
        
        # prepare input to model
        # input_mean_batch = torch.from_numpy(np.tile(input_mean, [input_batch.shape[0], 1]))
        mask_batch = torch.from_numpy(copula_generation(input_batch.shape[0]))

        # move input to DEVICE
        input_batch = input_batch.to(torch.float32).to(DEVICE)
        # input_mean_batch = input_mean_batch.to(torch.float32).to(DEVICE)
        mask_batch = mask_batch.to(torch.float32).to(DEVICE)
        label_batch = label_batch.to(torch.float32).to(DEVICE)

        # zero out gradient
        optimizer.zero_grad()

        # calculate loss
        output_batch = model_phase_2(input_batch, mask_batch)
        loss_1 = torch.mean(-torch.sum(label_batch * eval_utils.log(output_batch), dim=-1))
        loss_2 = torch.mean(model_phase_2.pi)
        loss_main = loss_1 + beta * loss_2

        # update parameters in model
        loss_main.backward()
        optimizer.step()

        # save loss info
        train_loss += loss_main
        train_step_count += 1
        model_update_count += 1

        # validate model every 1000 iterations
        if model_update_count % 1000 == 0:

            # display training info
            print(f"step: {model_update_count:06} | tr_loss: {train_loss / train_step_count:.6f}")
            train_loss = 0
            train_step_count = 0

            # model validation
            model_phase_2.eval()
            
            valid_loss = 0
            
            for input_batch, label_batch in valid_dataloader:

                # prepare input to model
                # input_mean_batch = torch.from_numpy(np.tile(input_mean, [input_batch.shape[0], 1]))
                mask_batch = torch.from_numpy(copula_generation(input_batch.shape[0]))

                # move input to DEVICE
                input_batch = input_batch.to(torch.float32).to(DEVICE)
                # input_mean_batch = input_mean_batch.to(torch.float32).to(DEVICE)
                mask_batch = mask_batch.to(torch.float32).to(DEVICE)
                label_batch = label_batch.to(torch.float32).to(DEVICE)

                # calculate loss
                output_batch = model_phase_2(input_batch, mask_batch)
                loss_1 = torch.mean(-torch.sum(label_batch * eval_utils.log(output_batch), dim=-1))
                loss_2 = torch.mean(model_phase_2.pi)
                loss_main = loss_1 + beta * loss_2

                # save loss info
                valid_loss += loss_main
            
            valid_loss /= len(valid_dataloader)

            print(f"step: {model_update_count:06} | va_loss: {valid_loss:.6f}")

            pi_statistics = [torch.mean(model_phase_2.pi[i*10:(i+1)*10]).item() for i in range(10)]
            print(f"step: {model_update_count:06} | {model_phase_2.pi[0]}, {model_phase_2.pi[10]}, {pi_statistics}")

            if valid_loss < min_valid_loss:
                print(f"Save model !")
                torch.save(model_phase_2, "checkpoint/model.pt")
                min_valid_loss = valid_loss
                valid_patience = 0
            else:
                valid_patience += 1
            
            if valid_patience >= 7:
                break

            model_phase_2.train()