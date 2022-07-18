from vanilla_gam import Generator, Discriminator_z2, Generator2, Discriminator2, Generator3, Discriminator3, Generator_z2
from utils import get_noise, data_sampler2,  save_models,  getstocks

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


# import yfinance as yf
#
# alsi=yf.download(
#     tickers="^J203.JO",
#     period="1y",
#     interval="1d"
# )
# alsi_daily_returns = alsi['Adj Close'].pct_change() + 1
# log_rets=np.log(alsi_daily_returns)
# S0=alsi['Adj Close'][0]
# # mu=alsi_daily_returns.mean()
# # sigma=alsi_daily_returns.std()
# mu=log_rets.mean()
# sigma=log_rets.std()


#todo wtf is this - export this to a module and update the other code sheet
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake samples.
    Parameters:
        crit: the critic model
        real: a batch of real samples
        fake: a batch of fake samples
        epsilon: a vector of the uniformly random proportions of real/fake per mixed sample
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed sample
    '''
    # Mix the samples together
    mixed_sample = real * epsilon + fake * (1 - epsilon)
    # Calculate the critic's scores on the mixed sample
    mixed_scores = crit(mixed_sample)
    # Take the gradient of the scores with respect to the sample
    gradient = torch.autograd.grad(
        inputs=mixed_sample,
        outputs=mixed_scores,
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of sample gradients, you calculate the magnitude of each sample's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed sample
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one sample
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean(pow(gradient_norm - torch.ones_like(gradient_norm), 2))
    return penalty

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake samples.
    Parameters:
        crit_fake_pred: the critic's scores of the fake samples
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real samples,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake samples
        crit_real_pred: the critic's scores of the real samples
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = torch.mean(crit_fake_pred)-torch.mean(crit_real_pred)+c_lambda*gp
    return crit_loss


# hyper parameters
num_iteration = 10000
num_gen = 1
num_crit = 5
lr = 0.0002
wd=0
batch_size = (128,20)
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
# gen=Generator()
crit=Discriminator_z2()
gen=Generator_z2()

# gen=Generator2()
# disc=Discriminator2()

# gen=Generator3()
# disc=Discriminator3()


gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=wd)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr,weight_decay=wd)
critic_loss = 0
critic_losses=[]
generator_loss = 0
generator_losses=[]

batch_size_target=(128,1)

#training
for iteration in range(num_iteration):
    for i in range(num_crit):
        crit_opt.zero_grad()

        target = data_sampler2(target_dist, target_param, batch_size_target)
        noise = data_sampler2(noise_dist, noise_param, batch_size)
        fakes = gen(noise).detach()
        pred_fake = crit(fakes)
        pred_real = crit(target)
        epsilon = torch.rand(len(target), 1, 1, 1, requires_grad=True)

        gradient = get_gradient(crit, target, fakes, epsilon)
        gp = gradient_penalty(gradient)
        crit_loss = get_crit_loss(pred_fake, pred_real, gp, 10)
        crit_loss.backward(retain_graph=True)
        crit_opt.step()

        critic_loss += crit_loss.item() / num_crit

        if i == num_crit - 1:
            critic_losses.append(critic_loss)

    for i in range(num_gen):
        gen_opt.zero_grad()
        noise = data_sampler2(noise_dist, noise_param, batch_size)
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

        target = data_sampler2(target_dist, target_param, batch_size_target)
        target = target.data.numpy().reshape(batch_size)
        noise = data_sampler2(noise_dist, noise_param, batch_size_target)
        transformed_noise = gen.forward(noise)
        transformed_noise = transformed_noise.data.numpy().reshape(batch_size)

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
noise = data_sampler2(noise_dist, noise_param, (100000,20))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)
rets=np.exp(transformed_noise)
np.quantile(rets,0.05)

sns.kdeplot(transformed_noise,fill=True)
plt.show()

kde = sm.nonparametric.KDEUnivariate(transformed_noise)
kde.fit()  # Estimate the densities

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Plot the histogram
ax.hist(
    transformed_noise,
    bins=20,
    density=True,
    label="Histogram from samples",
    zorder=5,
    edgecolor="k",
    alpha=0.5,
)

# Plot the KDE as fitted using the default arguments
ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)

# Plot the samples
ax.scatter(
    transformed_noise,
    np.abs(np.random.randn(transformed_noise.size)) / 40,
    marker="x",
    color="red",
    zorder=20,
    label="Samples",
    alpha=0.5,
)

ax.legend(loc="best")
ax.grid(True, zorder=-5)

plt.show()
