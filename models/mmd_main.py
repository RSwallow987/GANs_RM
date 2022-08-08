import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from mmd import mix_rbf_mmd2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vanilla_gam import GNet, Encoder, Decoder
from utils import data_sampler


# hyper parameters
num_iteration = 10000
num_gen = 1
num_enc_dec = 5
lr = 1e-3
batch_size = 128
target_dist = "gaussian"
target_param = (0, 0.02)
# target_dist = "uniform"
# target_param = (22, 24)
# target_dist = "cauchy"
# target_param = (23, 1)
noise_dist = "gaussian"
noise_param = (0., 1.)
# noise_dist = "uniform"
# noise_param = (-1, 1)

lambda_AE = 5. #as in paper
# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
print_int = 100

gen = GNet()
enc = Encoder()
dec = Decoder()
# lr = (1e-2, 1e-3, 1e-4)
criterion = nn.MSELoss()
gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
enc_optimizer = optim.Adam(enc.parameters(), lr=lr)
dec_optimizer = optim.Adam(dec.parameters(), lr=lr)


cum_dis_loss = 0
cum_gen_loss = 0
for iteration in range(num_iteration):
    for i in range(num_enc_dec):
        enc.zero_grad()
        dec.zero_grad()
        target = data_sampler(target_dist, target_param, batch_size)
        noise = data_sampler(noise_dist, noise_param, batch_size)
        encoded_target = enc.forward(target)
        decoded_target = dec.forward(encoded_target)
        L2_AE_target = (target - decoded_target).pow(2).mean()
        transformed_noise = gen.forward(noise)
        encoded_noise = enc.forward(transformed_noise)
        decoded_noise = dec.forward(encoded_noise)
        L2_AE_noise = (transformed_noise - decoded_noise).pow(2).mean()
        MMD = mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)
        MMD = F.relu(MMD)
        L_MMD_AE = -1 * (torch.sqrt(MMD)-lambda_AE*(L2_AE_noise+L2_AE_target))
        L_MMD_AE.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        cum_dis_loss = cum_dis_loss - L_MMD_AE.item()
    for i in range(num_gen):
        gen.zero_grad()
        target = data_sampler(target_dist, target_param, batch_size)
        noise = data_sampler(noise_dist, noise_param, batch_size)
        encoded_target = enc.forward(target)
        encoded_noise = enc.forward(gen.forward(noise))
        MMD = torch.sqrt(F.relu(mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)))
        MMD.backward()
        gen_optimizer.step()
        cum_gen_loss = cum_gen_loss + MMD.item()
    if iteration % print_int == 0 and iteration != 0:
        print('cum_dis_loss {}, cum_gen_loss {}'.format(cum_dis_loss/(print_int*num_enc_dec), cum_gen_loss/(print_int*num_gen)))
        cum_dis_loss = 0
        cum_gen_loss = 0
        target = data_sampler(target_dist, target_param, batch_size)
        target = target.data.numpy().reshape(batch_size)
        noise = data_sampler(noise_dist, noise_param, batch_size)
        transformed_noise = gen.forward(noise)
        transformed_noise = transformed_noise.data.numpy().reshape(batch_size)

        # plt.hist(target, 20)
        plt.hist(transformed_noise, 20)
        plt.show()

print("done")

noise = data_sampler(noise_dist, noise_param, 10000)
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy()
transformed_q=np.exp(np.quantile(transformed_noise,q=0.05))

target = data_sampler(target_dist, target_param, 10000)
target=target.data.numpy()
target_q=np.exp(np.quantile(target,q=0.05))

df=pd.DataFrame()

df['Actual']=pd.Series(target.flatten())
df['Generated']=pd.Series(transformed_noise.flatten())


fig=sns.kdeplot(df['Actual'], shade=True, color='r')
fig=sns.kdeplot(df['Generated'], shade=True, color='b')

plt.show()

