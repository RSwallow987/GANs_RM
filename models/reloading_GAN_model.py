from vanilla_gam import Generator_z2
from utils import get_noise, data_sampler2, save_models,  getstocks, gen_kde, image_name
import torch
import numpy as np
import matplotlib.pyplot as plt

gen = Generator_z2()
gen.load_state_dict(torch.load(f='../checkpoints/WGAN_ex125020-07-2022-07-01-59.pt', map_location='cpu'))

#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

noise = data_sampler2(noise_dist, noise_param, (100000,20))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)
rets=np.exp(transformed_noise)
np.quantile(rets,0.05)

x1,x2,q,k=gen_kde(transformed_noise)
plt.savefig(image_name("WGAN"))
plt.show()
print("Done")