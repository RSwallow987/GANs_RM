#VaR Backtesting Final Models
from vanilla_gam import GNet,Generator2
from utils import data_sampler2, gen_kde, image_name
import torch
import numpy as np
import matplotlib.pyplot as plt

#out of sample data
#MMD990009-08-2022-18-15-41.pt - GNet, 2.7% breeches - trained on 5,128 hist samples
#MMD10008-08-2022-08-16-28.pt - GNet , 13% breeches


data_set = data_sampler2("gaussian", (0.,0.02), (252,1))

gen = Generator2()
gen.load_state_dict(torch.load(f='../checkpoints/MMD_9800_09-08-2022-19-15-43.pt', map_location='cpu'))

#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

noise = data_sampler2(noise_dist, noise_param, (100000,1))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)

x1,x2 =gen_kde(transformed_noise)
plt.savefig(image_name("MMD"))
plt.show()

#Backtest
var95=np.quantile(transformed_noise,0.05)

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])

if num_breeches>len(k)*0.05:
    print("Out of Sample Breaches %:",num_breeches*100/len(k))
else:
    print("Adequate Model: Out of Sample Breeches %:",num_breeches*100/len(k))

#Backtest in sample
x=torch.load(f='../data quantiles/MMD_09-08-2022-19-16-06.pt')
x=x.reshape(-1).detach().numpy()
breeches_insample=np.where(x<var95)
num_breeches_in_sample=len(breeches_insample[0])

if num_breeches_in_sample>len(x)*0.05:
    print("In Sample Breaches %:",num_breeches_in_sample*100/len(x))
else:
    print("Adequate Model: In Sample Breeches %:",num_breeches_in_sample*100/len(x))

#Backtest Historical Model
mu=x.mean()
sig=x.std()
var95_hist=np.quantile(x,0.05)
breeches_hist=np.where(k<var95_hist)
num_breeches_hist=len(breeches_hist[0])

if num_breeches_hist>len(x)*0.05:
    print("In of Sample Breaches (Hist) %:",num_breeches_hist*100/len(x))
else:
    print("Adequate Model: In of Sample Breeches (Hist) %:",num_breeches_hist*100/len(x))
