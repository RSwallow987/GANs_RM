from vanilla_gam import Discriminator_z2, Generator_z2, Generator_Lz2,GNet, Discriminator, GeneratorLeak
from utils import data_sampler2,  save_models,  getstocks, gradient_penalty, get_gradient, get_gen_loss, get_crit_loss,gen_kde, save_hist, mixtureofnormals,get_crit_loss2

import torch
import matplotlib.pyplot as plt
import numpy as np


# hyper parameters
num_epochs = 10000
samps=124
num_gen = 1
num_crit = 5
lr = 0.001
wd=0
z=1

batch_size = (num_crit,samps)
noise_size=(samps,z)
target_dist = "gaussian"
# target_param = (23., 1.)
target_param=(0.,0.02)
display_step=250
# target_dist = "uniform"
# target_param = (22, 24)
# target_dist = "cauchy"
# target_param = (23, 1)
noise_dist = "gaussian"
noise_param = (0., 1.)
# noise_dist = "uniform"
# noise_param = (-1, 1)

#initialization
crit=Discriminator()
# gen=Generator_Lz2(z_dim=z)
gen=GeneratorLeak()

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=wd)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr,weight_decay=wd)
critic_loss = 0
critic_losses=[]
generator_loss = 0
generator_losses=[]

# data_set= data_sampler2(target_dist, target_param, batch_size)
b = (num_crit,samps)
# data_set= data_sampler2(target_dist, target_param, b)
weights=(0.5,0.5)
dist1=(1,0.2)
dist2=(2,0.2)
tot=num_crit*samps
data_set=mixtureofnormals(dist1,dist2,weights,tot,b)

#training
for iteration in range(num_epochs):
    for i in range(num_crit):
        crit_opt.zero_grad()
        target =data_set[i,:]
        target = torch.reshape(target, (samps, 1))
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        fakes = gen(noise).detach()
        pred_fake = crit(fakes)
        pred_real = crit(target)
        epsilon = torch.rand(len(target), 1, 1, 1, requires_grad=True)

        gradient = get_gradient(crit, target, fakes, epsilon)
        gp = gradient_penalty(gradient)
        crit_loss = get_crit_loss(pred_fake, pred_real, gp, 10)
        crit_loss = get_crit_loss2(pred_fake, pred_real, gp, 10)
        crit_loss.backward(retain_graph=True)
        crit_opt.step()

        critic_loss += crit_loss.item() / num_crit

        if i == num_crit - 1:
            critic_losses.append(critic_loss)

    for i in range(num_gen):
        gen_opt.zero_grad()
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        samples = gen(noise)
        pred = crit(samples)
        gen_loss = get_gen_loss(pred)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        generator_loss += gen_loss.item() / num_gen
        generator_losses.append(generator_loss)


    if iteration % display_step == 0 and iteration != 0:
        print('critic_loss {}, generator_loss {}'.format(critic_loss / (display_step * num_crit),
                                                         generator_loss / (display_step * num_gen)))

        critic_loss = 0
        generator_loss = 0

        save_models(gen, crit, str(iteration), gan_type="WGAN_ex")

        # target = data_sampler2(target_dist, target_param, batch_size_target)
        # target = target.data.numpy().reshape(batch_size_target)
        noise = data_sampler2(noise_dist, noise_param, noise_size)
        transformed_noise = gen.forward(noise)
        transformed_noise = transformed_noise.data.numpy().reshape((samps,1))

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

plt.plot(generator_losses, label='g_losses')
plt.plot(critic_losses, label='c_losses')
plt.legend()
plt.show()
print("Done")

#Testing
noise = data_sampler2(noise_dist, noise_param, (100000,z))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)
var95=np.quantile(transformed_noise,0.05)

#Backtest
x1,x2=gen_kde(transformed_noise.reshape(-1))
plt.show()

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])

if num_breeches>len(k)*0.05:
    print("Breached %:",num_breeches*100/len(k))
else:
    print("Adequate Model %:",num_breeches*100/len(k))

save_models(gen,crit,"final","WGAN")
save_hist(data_set, "WGAN")
