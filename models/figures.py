#VaR Backtesting Final Models
from vanilla_gam import Generator_Lz2, GNet, Generator2,Generator_z2
from utils import data_sampler2, gen_kde, image_name, mixtureofnormals
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

data_set = data_sampler2("gaussian", (0.,0.02), (1000,1))
# data_set=mixtureofnormals((1,0.2),(2,0.2),(0.5,0.5),252,(252,1))

z_mmd=20
z_ns=10
gen_mmd = Generator_z2()
gen_ns=Generator_Lz2(z_dim=z_ns)
gen_mmd.load_state_dict(torch.load(f='../checkpoints/MMD_3100_28-09-2022-16-39-22.pt', map_location='cpu'))
gen_ns.load_state_dict(torch.load(f='../checkpoints/NS_final_28-09-2022-20-00-58.pt', map_location='cpu'))
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
