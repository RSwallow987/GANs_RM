import torch
from torch import Tensor
import numpy as np
import yfinance as yf
import datetime
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    norm = torch.randn(n_samples, z_dim, device=device)  # outputs normal distribution N(0,1)
    uniform = torch.rand(n_samples, z_dim, device=device)  # outputs uniform distribution U(0,1)
    return norm, uniform

def data_sampler(dist_type, dist_param, batch_size=1):
    if dist_type=="gaussian":
        return Tensor(np.random.normal(dist_param[0], dist_param[1], (batch_size, 1))).requires_grad_()
    elif dist_type=="uniform":
        return Tensor(np.random.uniform(dist_param[0], dist_param[1], (batch_size, 1))).requires_grad_()
    elif dist_type=="cauchy":
        return dist_param[1] * Tensor(np.random.standard_cauchy((batch_size, 1))) + 23.

def data_sampler2(dist_type, dist_param, batch_size):
    if dist_type=="gaussian":
        return Tensor(np.random.normal(dist_param[0], dist_param[1], batch_size)).requires_grad_()
    elif dist_type=="uniform":
        return Tensor(np.random.uniform(dist_param[0], dist_param[1], (batch_size[0], batch_size[1]))).requires_grad_()
    elif dist_type=="cauchy":
        return dist_param[1] * Tensor(np.random.standard_cauchy(((batch_size[0], batch_size[1])))) + 23.

def save_models(generator, discriminator, epoch, gan_type):
  """ Save models at specific point in time. """
  # at specified directory
  # get current date and time
  x = datetime.datetime.now()
  file_name = r"C:\Users\rswal\PycharmProjects\Thesis\checkpoints\\" + gan_type + "_" + epoch+ "_" + x.strftime('%d-%m-%Y-%H-%M-%S.pt')
  with open(file_name, 'w') as fp:
      print('created', file_name)
  torch.save(generator.state_dict(),file_name)

def save_hist(tensor,gan_type):
  """ Save models at specific point in time. """
  # at specified directory
  # get current date and time
  x = datetime.datetime.now()
  file_name = r"C:\Users\rswal\PycharmProjects\Thesis\data quantiles\\" + gan_type + "_" + x.strftime('%d-%m-%Y-%H-%M-%S.pt')
  with open(file_name, 'w') as fp:
      print('created', file_name)
  torch.save(tensor,file_name)


# Import data - only for multiple stocks
def getstocks(stocks, start, end):
    stockdf = yf.download(
        tickers = stocks,
        start=start,
        end=end,
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = False,
        threads = True,
        proxy = None
    )
    stockdf= stockdf.iloc[:, stockdf.columns.get_level_values(1)=='Close']
    returns = stockdf.pct_change().dropna()
    current=stockdf.tail(1).T
    return returns, current

def rolling_stats(returns, window):
    mu=returns.rolling(window=window, center=False, min_periods=window).mean()
    sigma=returns.rolling(window=window, center=False, min_periods=window).std()
    cov=returns.rolling(window=window, center=False, min_periods=window, method='table').cov()
    return mu, sigma, cov

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

def gen_kde(transformed_noise):

    fig1=sns.kdeplot(transformed_noise, fill=True)

    kde = sm.nonparametric.KDEUnivariate(transformed_noise)
    kde.fit()  # Estimate the densities
    # q=kde.icdf[:]
    # d=kde.cdf[:]

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

    return fig1, ax

def image_name(gan_type):
  """ Save models at specific point in time. """
  # at specified directory
  # get current date and time
  x = datetime.datetime.now()
  file_name = r"C:\Users\rswal\PycharmProjects\Thesis\images\\" +gan_type+ x.strftime('%d-%m-%Y-%H-%M-%S.png')
  # with open(file_name, 'w') as fp:
  #     print('created', file_name)
  return file_name

def moments_test(real,fake):
    mu=math.sqrt((real.mean-fake.mean)**2)
    v=math.sqrt((real.variance-fake.variance)**2)
    sk=math.sqrt((real.skewness-fake.skewness)**2)
    k=math.sqrt((real.kurtosis-fake.kurtosis)**2)

    return mu,v,sk,k