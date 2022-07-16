import torch
from torch import Tensor
import numpy as np
import yfinance as yf
import datetime

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

def save_models(generator, discriminator, epoch):
  """ Save models at specific point in time. """
  # at specified directory
  # get current date and time
  x = datetime.datetime.now()
  file_name = r"C:\Users\rswal\PycharmProjects\Thesis\checkpoints\\" + x.strftime('%d-%m-%Y-%H-%M-%S.pt')
  with open(file_name, 'w') as fp:
      print('created', file_name)
  torch.save(generator.state_dict(),file_name)
  # torch.save(discriminator.state_dict(), f'./checkpoints/discriminator_{epoch}.pt')

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

#todo ewma
#todo log returns ?
