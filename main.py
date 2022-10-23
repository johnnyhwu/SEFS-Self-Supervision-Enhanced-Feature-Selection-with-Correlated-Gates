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
# phase 1: self-supervised leaning
model = model_utils.Phase_1_Model()
model.to(DEVICE)
model.train()

train_dataset = data_uitls.Phase_1_Dataset(X=X_train_phase_1[1000:])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = data_uitls.Phase_1_Dataset(X=X_train_phase_1[:1000])
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)



optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer_1 = optim.Adam(model.enc.parameters(), lr=1e-4)
optimizer_2 = optim.Adam(model.dec_x.parameters(), lr=1e-4)
optimizer_3 = optim.Adam(model.dec_m.parameters(), lr=1e-4)

#%%
input_mean = np.mean(X_train_phase_1, axis=0, keepdims=True)
total_epoch = 10000
fprob_threshold = 0.5
alpha = 10.0
model_update_count = 0

#%%
for epoch_idx in range(total_epoch):

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
        # optimizer_1.zero_grad()
        # optimizer_2.zero_grad()
        # optimizer_3.zero_grad()
        optimizer.zero_grad()

        # calculate loss
        output_1_batch, output_2_batch = model(input_batch, input_mean_batch, mask_batch)
        loss_recon_x = torch.mean(torch.sum((output_1_batch - label_batch)**2, axis=1))
        loss_recon_m = torch.mean(-torch.sum(mask_batch * eval_utils.log(output_2_batch) + (1. - mask_batch) * eval_utils.log(1. - output_2_batch), axis=1))            
        loss_main = loss_recon_x + alpha * loss_recon_m

        # update parameters in model
        loss_main.backward()
        # optimizer_1.step()
        # optimizer_2.step()
        # optimizer_3.step()
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
            print(f"{'='*10} step: {model_update_count} {'='*10}")
            print(f"train_loss_recon_x: {train_loss_recon_x / train_step_count}")
            print(f"train_loss_recon_m: {train_loss_recon_m / train_step_count}")
            print(f"train_loss: {train_loss / train_step_count}")
            train_loss_recon_x = 0
            train_loss_recon_m = 0
            train_loss = 0
            train_step_count = 0

            # model validation
            model.eval()
            
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
                output_1_batch, output_2_batch = model(input_batch, input_mean_batch, mask_batch)
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

            print(f"valid_loss_recon_x: {valid_loss_recon_x}")
            print(f"valid_loss_recon_m: {valid_loss_recon_m}")
            print(f"valid_loss: {valid_loss}")

            model.train()

exit()

# %% [markdown]
# # STEP2: SUPERVISION PHASE

# %%
reg_scale      = 0. 

num_layers_p   = 1
h_dim_p        = 100


input_dims = {
    'x_dim': x_dim,
    'z_dim': z_dim,
    'y_dim': y_dim,
    'y_type': y_type
} 


network_settings = {
    'h_dim_e': h_dim_e,
    'num_layers_e': num_layers_e,
    'h_dim_p': h_dim_p,
    'num_layers_p': num_layers_p,
    
    'fc_activate_fn_e': tf.nn.relu, 
    'fc_activate_fn_p': tf.nn.relu, 
    
    'reg_scale': reg_scale
}

# %%
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
# config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
model = SEFS_S_Phase(sess, "feature_selection", input_dims, network_settings)

# %%
step_size      = 1000
iteration      = 500000

# %%
mb_size        = 32
mb_size        = min(mb_size, np.shape(tr_X)[0])
 
learning_rate  = 1e-4

keep_prob      = 1.0 
lmbda          = 1.0 

# %%
sess.run(tf.global_variables_initializer())
saver       = tf.train.Saver()

# %%
pretrained_encoder = np.load(save_path + 'model_pretrained_encoder.npz', allow_pickle=True)

for i in range(len(list(pretrained_encoder))):
    sess.run(tf.assign(model.vars_encoder[i], pretrained_encoder[list(pretrained_encoder)[i]]))

# %%
va_X        = np.copy(tr_X)
va_Y_onehot = np.copy(tr_Y_onehot)

# %%
print('=============================================')
print('Start Feature Selection .... OUT_ITR {}, LAMBDA {}, KEEP PROB {}'.format(out_itr, lmbda, keep_prob))
print('=============================================')

avg_loss      = 0.
avg_loss_m0   = 0.   

va_avg_loss      = 0.
va_avg_loss_m0   = 0.

max_auc      = 0.    
min_loss     = 1e+8

max_flag     = 20
stop_flag    = 0

num_selected_curr = 0
num_selected_prev = 0


for itr in range(iteration):
    x_mb, y_mb     = f_get_minibatch(min(mb_size, np.shape(tr_X)[0]), tr_X, tr_Y_onehot)
    x2_mb          = np.tile(x_mean, [np.shape(x_mb)[0], 1])
    q_mb           = copula_generation(mb_size)
    
    _, tmp_loss, tmp_loss_m0  = model.train_finetune(x_=x_mb, x_bar_=x2_mb, y_=y_mb, q_=q_mb, lmbda_=lmbda, lr_train_=learning_rate, k_prob_=keep_prob)
    avg_loss      += tmp_loss/step_size
    avg_loss_m0   += tmp_loss_m0/step_size
    

    tmp_loss, tmp_loss_m0     = model.get_loss(x_=x_mb, x_bar_=x2_mb, y_=y_mb, q_=q_mb, lmbda_=lmbda)
    
    va_avg_loss      += tmp_loss/step_size
    va_avg_loss_m0   += tmp_loss_m0/step_size    
                
    
    if (itr+1)%step_size == 0:
        stop_flag  += 1

        tmp_mask = (sess.run(model.pi) > 0.5).astype(float)
        q_mb     = copula_generation(np.shape(va_X)[0])
        
        tmp_y    = model.predict(x_=va_X, x_bar_=np.tile(x_mean, [np.shape(va_X)[0], 1]), q_=q_mb)
        tmp_y2   = model.predict_final(x_=va_X, x_bar_=np.tile(x_mean, [np.shape(va_X)[0], 1]), m_=tmp_mask)
        
        va_auc, va_apc   = cal_metrics(va_Y_onehot, tmp_y)
        va_auc2, va_apc2 = cal_metrics(va_Y_onehot, tmp_y2)

        print("ITR {:05d}  | TR: loss={:.3f} loss_m0={:.3f}  | VA: loss={:.3f} loss_m0={:.3f} AUC:{:.3f}, AUC_Selected:{:.3f}".format(
            itr+1, avg_loss, avg_loss_m0, va_avg_loss, va_avg_loss_m0, va_auc, va_auc2
        ))
        


        if va_avg_loss < min_loss:
            print('saved...')
            saver.save(sess, save_path + 'sefs_trained')
            
            min_loss  = va_avg_loss
            
            stop_flag = 0


        avg_loss      = 0.
        avg_loss_m0   = 0.   

        va_avg_loss      = 0.
        va_avg_loss_m0   = 0.

        if stop_flag >= max_flag:
            break

# %%
feature_importance = sess.run(model.pi)

# %%
plt.bar(range(x_dim), feature_importance) 
plt.bar(range(0, x_dim, blocksize), feature_importance[range(0, x_dim, blocksize)])  # x0, x10, x20, x30, ...


plt.xlabel('Feature Number', fontsize=12)
plt.ylabel('Feature Importance', fontsize=12)
plt.show()
plt.close()


