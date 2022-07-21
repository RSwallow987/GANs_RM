from vanilla_gam import Generator, Discriminator, Generator2, Discriminator2, Generator3, Discriminator3, Generator_z, Generator_z2,Discriminator_z2
from utils import get_noise, data_sampler, save_models,  getstocks, gen_kde, data_sampler2

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# hyper parameters
num_epochs = 10000
samps=200
num_gen = 1
num_disc = 5
lr = 2e-3
batch_size = (num_disc,samps)
noise_size=(samps,20)
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
gan_type="vanilla"


# gen=Generator()
disc=Discriminator_z2()
gen=Generator_z2()

# gen=Generator2()
# disc=Discriminator2()

# gen=Generator3()
# disc=Discriminator3()

criterion=nn.BCEWithLogitsLoss()
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
discriminator_loss = 0
discriminator_losses=[]
generator_loss = 0
generator_losses=[]


#training
data_set= data_sampler2(target_dist, target_param, batch_size)

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
        # target = data_sampler(target_dist, target_param, batch_size)
        # target = target.data.numpy().reshape(batch_size)
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
        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)) #Assuming normal distribution
        ax.plot(bins, y, '--')
        ax.set_xlabel('')
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of Generated Data')
        fig.tight_layout()
        plt.show()

plt.plot(generator_losses, label='g_losses')
plt.plot(discriminator_losses, label='d_losses')
plt.legend()
plt.show()

#Testing
noise = data_sampler2(noise_dist, noise_param, (100000,20))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)
# rets=np.exp(transformed_noise)
var95=np.quantile(transformed_noise,0.05)

x1,x2=gen_kde(transformed_noise)
plt.show()

print("Done")

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])