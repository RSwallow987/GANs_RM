import pandas as pd
import numpy as np

window=250
alpha=0.05

rmfi=pd.read_excel('../data/Practical 1 - Introduction to VaR - Solution.xlsm', sheet_name='Data', usecols="A:K", skiprows=1)
rmfi=rmfi.set_index('Date')
rmfi_pct=rmfi.pct_change().dropna()
units=np.repeat(50000,10)
price=rmfi.tail(1).T
price['units']=units
price['values']=price.iloc[:,0]*price.iloc[:,1]
scenarios=rmfi_pct
current_portfolio=price['values']

def multindex_iloc(df, index):
    label = df.index.levels[0][index]
    return df.iloc[df.index.get_loc(label)]

returns = scenarios + 1
cov = returns.rolling(window=window, center=False, min_periods=window, method='table').cov()
mu = returns.rolling(window=window, center=False, min_periods=window).mean()
v = np.empty((len(returns), 1))

for j in range(0,(len(returns) - 1),1):
    random_norms = np.random.standard_normal(size=(100, 10))
    x = np.empty((len(random_norms), 10))
    per = np.empty((len(random_norms), 1))
    covar = multindex_iloc(cov, j)
    mus=mu.iloc[j, :].T
    if covar.reset_index().isna().iloc[0, 3]== False :
         for i in range(0,(len(random_norms)),1):
            x[i] = ((((np.linalg.cholesky(covar).dot(random_norms[i, :])) + mus)))
            rets_shares = (np.linalg.cholesky(covar).dot(random_norms[i, :])) +mus
            per[i] = rets_shares.dot(current_portfolio) / current_portfolio.sum() - 1
         v[j] = np.quantile(per, q=alpha)
    else :
        v[j]=0
