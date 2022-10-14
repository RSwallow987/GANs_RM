from vanilla_gam import Generator, Discriminator, Generator2, Discriminator2, Generator3, Discriminator3, Generator_z, Generator_z2,Discriminator_z2, GNet, Encoder, Generator_Lz2
from utils import get_noise, data_sampler, save_models,  getstocks, gen_kde, data_sampler2,save_hist, mixtureofnormals3

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# hyper parameters
num_epochs = 10000
samps=128*2
num_gen = 1
num_disc = 5
lr = 1e-3
z=10
batch_size = (num_disc,samps)
noise_size=(samps,z)
target_dist = "gaussian"
# target_param = (23., 1.)
target_param=(0.0,0.02)
display_step=500
# target_dist = "uniform"
# target_param = (22, 24)
# target_dist = "cauchy"
# target_param = (23, 1)
noise_dist = "gaussian"
noise_param = (0., 1.)
# noise_dist = "uniform"
# noise_param = (-1, 1)

#initialization
#_______________________________CHANGE____________________________#
gan_type="NS"

disc=Discriminator_z2()
gen=Generator_Lz2(z_dim=z)

criterion=nn.BCEWithLogitsLoss()
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
discriminator_loss = 0
discriminator_losses=[]
generator_loss = 0
generator_losses=[]

#training
# data_set= data_sampler2(target_dist, target_param, batch_size)
b = (num_disc,samps)
# data_set= data_sampler2(target_dist, target_param, b)
weights=(0.07,0.05,0.88)
dist1=(0.0282,0.0099)
dist2=(-0.0315,0.01356)
dist3=(-0.0001,0.0092)
tot=num_disc*samps
data_set=mixtureofnormals3(dist1,dist2,dist3,weights,tot,b)



for iteration in range(num_epochs):
    for i in range(num_disc):
        disc_opt.zero_grad()
        target=data_set[i,:]
        target=torch.reshape(target, (samps, 1))
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        fakes = gen(noise).detach()
        pred_fake = disc(fakes)
        zeros = torch.zeros_like(pred_fake)
        fake_loss = criterion(pred_fake, zeros)
        pred_real = disc(target)
        ones = torch.ones_like(pred_real)
        real_loss = criterion(pred_real, ones)
        disc_loss = (fake_loss + real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        discriminator_loss += disc_loss.item() / num_disc

        if i == num_disc-1:
            discriminator_losses.append(discriminator_loss)

    for i in range(num_gen):
        gen_opt.zero_grad()
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        samples = gen(noise)
        pred = disc(samples)
        gen_loss = criterion(pred, torch.ones_like(pred))
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        generator_loss += gen_loss.item()/num_gen
        generator_losses.append(generator_loss)


    if iteration % display_step == 0 and iteration != 0:
        print('discriminator_loss {}, generator_loss {}'.format(discriminator_loss/(display_step*num_disc), generator_loss/(display_step*num_gen)))
        save_models(gen,disc,str(iteration),gan_type)
        discriminator_loss = 0
        generator_loss = 0
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        transformed_noise = gen.forward(noise)
        transformed_noise = transformed_noise.data.numpy().reshape((samps,1))
        #Visualization
        mu = transformed_noise.mean()
        sigma = transformed_noise.std()  # standard deviation of distribution
        x = transformed_noise
        num_bins=50
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(x, num_bins, density=True)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)) #Assuming normal distribution
        ax.plot(bins, y, '--')
        ax.set_xlabel('')
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of Generated Data')
        fig.tight_layout()
        plt.show()

save_models(gen,disc,str(iteration),gan_type)
plt.plot(generator_losses, label='g_losses')
plt.plot(discriminator_losses, label='d_losses')
plt.legend()
plt.show()
print("Done")

#Testing
noise = data_sampler2(noise_dist, noise_param, (100000,z))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)
var95=np.quantile(transformed_noise,0.05)

x1,x2=gen_kde(transformed_noise)
plt.show()

print("Done")

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])

if num_breeches>len(k)*0.05:
    print("Breached %:",num_breeches*100/len(k))
else:
    print("Adequate Model %:",num_breeches*100/len(k))

save_models(gen,disc,"final",gan_type)
save_hist(data_set, "NS")