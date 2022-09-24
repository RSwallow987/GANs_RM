#VaR Backtesting Final Models
from vanilla_gam import Generator_z2, GNet, Generator2
from utils import data_sampler2, gen_kde, image_name
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
data_set = data_sampler2("gaussian", (0.,0.02), (252,1))

z_mmd=1
z_ns=20
gen_mmd = Generator2()
gen_ns=Generator_z2()
gen_mmd.load_state_dict(torch.load(f='../checkpoints/MMD_9800_09-08-2022-19-15-43.pt', map_location='cpu'))
gen_ns.load_state_dict(torch.load(f='../checkpoints/NS_9500_15-08-2022-20-26-28.pt', map_location='cpu'))
#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

#NS
noise_ns = data_sampler2(noise_dist, noise_param, (100000,z_ns))
transformed_noise_ns = gen_ns.forward(noise_ns)
transformed_noise_ns = transformed_noise_ns.data.numpy().reshape(100000)

#MMD
noise_mmd = data_sampler2(noise_dist, noise_param, (100000,z_mmd))
transformed_noise_mmd = gen_mmd.forward(noise_mmd)
transformed_noise_mmd= transformed_noise_mmd.data.numpy().reshape(100000)

k=data_set.reshape(-1).detach().numpy()

#Visualize
df=pd.DataFrame()

df['Actual']=pd.Series(k.flatten())
df['Generated MMD']=pd.Series(transformed_noise_mmd.flatten())
df['Generated NS']=pd.Series(transformed_noise_ns.flatten())
fig=sns.kdeplot(df['Actual'], shade=True, color='r')
fig=sns.kdeplot(df['Generated MMD'], shade=False, color='b')
fig=sns.kdeplot(df['Generated NS'], shade=False, color='g')
plt.legend(labels=["Actual","Generated MMD", "Generated NS"])
plt.xlabel("")
plt.show()
