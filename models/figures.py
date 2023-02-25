#VaR Backtesting Final Models
from vanilla_gam import Generator_Lz2, GNet, Generator2,Generator_z2
from utils import data_sampler2, gen_kde, image_name, mixtureofnormals, mixtureofnormals3
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

data_set = data_sampler2("gaussian", (23.,1), (1000,1))
# data_set=mixtureofnormals((1,0.2),(2,0.2),(0.5,0.5),252,(252,1))

num=3
# # data_set=mixtureofnormals((1,0.2),(2,0.2),(0.5,0.5),252,(252,1))
weights=(0.07,0.05,0.88)
dist1=(0.0282,0.0099)
dist2=(-0.0315,0.01356)
dist3=(-0.0001,0.0092)
tot=250
data_set=mixtureofnormals3(dist1,dist2,dist3,weights,tot,(tot,1))

z_mmd=20
z_ns=10
z_wgan=10

gen_mmd = Generator_z2(z_dim=z_mmd)
gen_ns=Generator_Lz2(z_dim=z_ns)
gen_wgan=Generator_z2(z_dim=z_wgan)

gen_mmd.load_state_dict(torch.load(f='../checkpoints/MMD_final_13-10-2022-07-39-39.pt', map_location='cpu'))
gen_ns.load_state_dict(torch.load(f='../checkpoints/NS_final_12-10-2022-06-18-08.pt', map_location='cpu'))
gen_wgan.load_state_dict(torch.load(f='../checkpoints/WGAN_final_23-02-2023-18-07-05.pt', map_location='cpu'))

# gen_ns320=Generator_Lz2(z_dim=z_ns)
# gen_ns640=Generator_Lz2(z_dim=z_ns)
# gen_ns1280=Generator_Lz2(z_dim=z_ns)
#
# gen_ns320.load_state_dict(torch.load(f='../checkpoints/NS_23_7000_13-02-2023-20-02-47.pt', map_location='cpu'))
# gen_ns640.load_state_dict(torch.load(f='../checkpoints/NS_23_2500_13-02-2023-19-19-09.pt', map_location='cpu'))
# gen_ns1280.load_state_dict(torch.load(f='../checkpoints/NS_23_2500_13-02-2023-19-47-29.pt', map_location='cpu'))

#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

# #NS
noise_ns = data_sampler2(noise_dist, noise_param, (100000,z_ns))
transformed_noise_ns = gen_ns.forward(noise_ns)
transformed_noise_ns = transformed_noise_ns.data.numpy().reshape(100000)

# #MMD
noise_mmd = data_sampler2(noise_dist, noise_param, (100000,z_mmd))
transformed_noise_mmd = gen_mmd.forward(noise_mmd)
transformed_noise_mmd= transformed_noise_mmd.data.numpy().reshape(100000)

# #WGAN
noise_wgan = data_sampler2(noise_dist, noise_param, (100000,z_wgan))
transformed_noise_wgan= gen_wgan.forward(noise_wgan)
transformed_noise_wgan= transformed_noise_wgan.data.numpy().reshape(100000)


#NS1
# noise_ns = data_sampler2(noise_dist, noise_param, (100000,z_ns))
# transformed_noise_ns = gen_ns320.forward(noise_ns)
# transformed_noise_ns = transformed_noise_ns.data.numpy().reshape(100000)
#
# #NS2
# noise_mmd = data_sampler2(noise_dist, noise_param, (100000,z_mmd))
# transformed_noise_640= gen_ns640.forward(noise_mmd)
# transformed_noise_640= transformed_noise_640.data.numpy().reshape(100000)
#
# #NS3
# noise_wgan = data_sampler2(noise_dist, noise_param, (100000,z_wgan))
# transformed_noise_1280= gen_ns1280.forward(noise_wgan)
# transformed_noise_1280= transformed_noise_1280.data.numpy().reshape(100000)

k=data_set.reshape(-1).detach().numpy()

#Visualize
df=pd.DataFrame()

fig, axes = plt.subplots(2, 1, figsize=(8,10))
#
df['Actual']=pd.Series(k.flatten())
df['Generated MMD']=pd.Series(transformed_noise_mmd.flatten())
df['Generated NS']=pd.Series(transformed_noise_ns.flatten())
df['Generated WGAN']=pd.Series(transformed_noise_wgan.flatten())
fig=sns.kdeplot(df['Actual'], shade=True, color='r',ax=axes[0])
fig=sns.kdeplot(df['Generated MMD'], shade=False, color='b',ax=axes[0])
fig=sns.kdeplot(df['Generated NS'], shade=False, color='g',ax=axes[0])
fig=sns.kdeplot(df['Generated WGAN'], shade=False, color='black',ax=axes[0])
axes[0].set_xlabel("Value")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Cumulative Density")
axes[0].legend(title=" ", labels=["Actual","Generated MMD","Generated NS","Generated WGAN",])
plt_df=df.melt()
fig3=sns.ecdfplot(data=plt_df, x='value',hue='variable')
axes[1].legend(title=" ", labels=["Actual","Generated MMD","Generated WGAN","Generated NS"])
plt.show()

# plt.figure(figsize=(10,8))

# df['Actual']=pd.Series(k.flatten())
# df['Generated 320']=pd.Series(transformed_noise_ns.flatten())
# df['Generated 640']=pd.Series(transformed_noise_640.flatten())
# df['Generated 1280']=pd.Series(transformed_noise_1280.flatten())
# # fig=sns.kdeplot(df['Actual'], shade=True, color='r',ax=axes[0])
# # fig=sns.kdeplot(df['Generated 320'], shade=False, color='b',ax=axes[0])
# # fig=sns.kdeplot(df['Generated 640'], shade=False, color='g',ax=axes[0])
# # # fig=sns.kdeplot(df['Generated 1280'], shade=False, color='black',ax=axes[0])
# # axes[0].set_xlabel("Value")
# plt.xlabel("Value")
# plt.ylabel("Cumulative Density")
# # axes[0].legend(title=" ", labels=["Actual","Generated 320","Generated 640","Generated 1280"])
# plt_df=df.melt()
# fig3=sns.ecdfplot(data=plt_df, x='value',hue='variable')
# plt.legend(title=" ", labels=["Actual","Generated 320","Generated 640","Generated 1280"])
# plt.show()
