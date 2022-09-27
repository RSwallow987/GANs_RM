import pandas as pd
import numpy as np
from scipy.stats import norm

def hist_var(current_portfolio, scenarios, window,alpha ):
    '''
           Function for completing the historical var : Given the current unit price of each share,
           historical daily closing price percentage changes, rolling window and confidence interval
           Parameters:
               current_portfolio: Most recent prices for stocks in portfolio
               scenarios: Historical daily percentage changes in closing proce for each stock in portfolio
               window: rolling statistic window (business days)
               alpha:n confidence interval
            Returns:
                vars: percentile vars (daily chnages not returns)
                vars_nom: nominal value vars
           '''
    returns=scenarios+1
    portfolio = returns.dot(current_portfolio)
    portfolio_returns=portfolio/current_portfolio.sum() -1
    # x = portfolio_returns.to_numpy()
    # var1,var5= np.quantile(x,q=[0.01,0.05])
    vars=portfolio_returns.rolling(window=window, center=False, min_periods=window).quantile(quantile=alpha,interpolation='lower')
    vars_nom=(vars)*-1*current_portfolio.sum()
    return vars, vars_nom

def norm_linear_var(current_portfolio, scenarios, window, alpha):
    '''
       Function for completing the normal linear var : Given the current unit price of each share,
       historical daily closing price percentage changes, rolling window and confidence interval
       Parameters:
           current_portfolio: Most recent prices for stocks in portfolio
           scenarios: Historical daily percentage changes in closing proce for each stock in portfolio
           window: rolling statistic window (business days)
           alpha:n confidence interval
        Returns:
            vars_nom: nominal portfolio level var
            vars: quantiles (? )

    '''
    returns = scenarios+1
    portfolio=returns.dot(current_portfolio)
    portfolio_returns = portfolio / current_portfolio.sum() - 1
    mu=portfolio_returns.rolling(window=window, center=False, min_periods=window).mean()
    sigma=portfolio_returns.rolling(window=window, center=False, min_periods=window).std()
    vars_nom= (norm.ppf(1 - alpha) * sigma - mu) * current_portfolio.sum()
    vars=(mu-sigma*norm.ppf(alpha))
    return vars, vars_nom

def monte_carlo_var(current_portfolio,scenarios, window, alpha):
    '''
         Function for completing the monte carlo var : Given the current unit price of each share,
         historical daily closing price percentage changes, rolling window and confidence interval
         Parameters:
             current_portfolio: Most recent prices for stocks in portfolio
             scenarios: Historical daily percentage changes in closing proce for each stock in portfolio
             window: rolling statistic window (business days)
             alpha:n confidence interval
          Returns:
              v: quantiles
              vars_nom: nominal var
     '''
    returns = scenarios + 1
    cov = returns.rolling(window=window, center=False, min_periods=window, method='table').cov()
    mu = returns.rolling(window=window, center=False, min_periods=window).mean()
    v = np.empty((len(returns), 1))

    for j in range(0, (len(returns) - 1), 1):
        random_norms = np.random.standard_normal(size=(100, returns.shape[1]))
        per = np.empty((len(random_norms), 1))
        covar = multindex_iloc(cov, j)
        mus = mu.iloc[j, :].T
        if covar.reset_index().isna().iloc[0, (covar.shape[1]+1)] == False:
            for i in range(0, (len(random_norms)), 1):
                rets_shares = (np.linalg.cholesky(covar).dot(random_norms[i, :])) + mus
                per[i] = rets_shares.dot(current_portfolio) / current_portfolio.sum() - 1
            v[j] = np.quantile(per, q=alpha)
        else:
            v[j] = 0

        vars_nom=-v*current_portfolio.sum()

    return v, vars_nom

def multindex_iloc(df, index):
    label = df.index.levels[0][index]
    return df.iloc[df.index.get_loc(label)]

