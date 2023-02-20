import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mmd import mix_rbf_mmd2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vanilla_gam import GNet, Encoder, Decoder,Generator2, Generator_z2
from utils import data_sampler2, data_sampler, save_models, gen_kde,save_hist,mixtureofnormals3

# hyper parameters
num_epochs = 10000
num_gen = 1
num_enc_dec = 5
lr = 1e-3 # lr = (1e-2, 1e-3, 1e-4)
z=20
samp=128*2
batch_size = (samp,z)

# Dist1
target_dist = "gaussian"
target_param = (23., 1.)
# noise_dist = "gaussian"
# noise_param = (0., 1.)

# #Dist2
# target_dist = "lognorm"
# target_param = (23., 1.)
noise_dist = "uniform"
noise_param = (-1, 1)

#Dist 3
# weights=(0.07,0.05,0.88)
# dist1=(0.0282,0.0099)
# dist2=(-0.0315,0.01356)
# dist3=(-0.0001,0.0092)
# tot=num_disc*samps
# data_set=mixtureofnormals3(dist1,dist2,dist3,weights,tot,b)

lambda_AE = 8. #as in paper

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]
print_int = 100

# gen = GNet()
# gen=Generator2()
gen=Generator_z2()
enc = Encoder()
dec = Decoder()

criterion = nn.MSELoss()
gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
enc_optimizer = optim.Adam(enc.parameters(), lr=lr)
dec_optimizer = optim.Adam(dec.parameters(), lr=lr)


b = (num_enc_dec,samp)
data_set= data_sampler2(target_dist, target_param, b)

cum_dis_loss = 0
cum_gen_loss = 0
for iteration in range(num_epochs):
    for i in range(num_enc_dec):
        enc.zero_grad()
        dec.zero_grad()
        target = data_set[i, :]
        target = torch.reshape(target, (samp, 1))
        # noise = data_sampler(noise_dist, noise_param, batch_size)
        noise = data_sampler2(noise_dist, noise_param, batch_size)
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
        target = data_set[i, :]
        target = torch.reshape(target, (samp, 1))
        noise = data_sampler2(noise_dist, noise_param, batch_size)
        encoded_target = enc.forward(target)
        encoded_noise = enc.forward(gen.forward(noise))
        MMD = torch.sqrt(F.relu(mix_rbf_mmd2(encoded_target, encoded_noise, sigma_list)))
        MMD.backward()
        gen_optimizer.step()
        cum_gen_loss = cum_gen_loss + MMD.item()
    if iteration % print_int == 0 and iteration != 0:
        print('cum_dis_loss {}, cum_gen_loss {}'.format(cum_dis_loss/(print_int*num_enc_dec), cum_gen_loss/(print_int*num_gen)))
        save_models(gen, enc, str(iteration), "MMD")
        cum_dis_loss = 0
        cum_gen_loss = 0
        noise = data_sampler2(noise_dist, noise_param, batch_size)
        transformed_noise = gen.forward(noise)
        transformed_noise = transformed_noise.data.numpy().reshape((samp, 1))

        # Visualization
        mu = transformed_noise.mean()
        sigma = transformed_noise.std()  # standard deviation of distribution
        x = transformed_noise
        num_bins = 50

        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(x, num_bins, density=True)
        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Assuming normal distribution
        ax.plot(bins, y, '--')
        ax.set_xlabel('')
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of Generated Data')
        fig.tight_layout()
        plt.show()
print("Done")

noise = data_sampler2(noise_dist, noise_param, (10000,z))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy()

target = data_sampler(target_dist, target_param, 10000)
# target=mixtureofnormals(dist1,dist2,weights,tot,b)
# data_set=mixtureofnormals3(dist1,dist2,dist3, weights,10000,(10000,1))
target=target.data.numpy()

df=pd.DataFrame()

df['Actual']=pd.Series(target.flatten())
df['Generated']=pd.Series(transformed_noise.flatten())


fig=sns.kdeplot(df['Actual'], shade=True, color='r')
fig=sns.kdeplot(df['Generated'], shade=True, color='b')

plt.show()

#Backtest
var95=np.quantile(transformed_noise,0.05)
x1,x2=gen_kde(transformed_noise.reshape(-1))
plt.show()

print("Done")

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])

if num_breeches>len(k)*0.05:
    print("Breached %:",num_breeches*100/len(k))
else:
    print("Adequate Model %:",num_breeches*100/len(k))

#Hist VaR
save_hist(data_set, "MMD") #Save data set for hist VaR Model.
save_models(gen,enc,"final","MMD")