from VaR import hist_var, norm_linear_var, monte_carlo_var
from utils import getstocks,data_sampler
import numpy as np

df, current=getstocks("FSR.JO MTN.JO AGL.JO BVT.JO IMP.JO NPN.JO SBK.JO", '2002-04-09', '2012-03-30')
df2=data_sampler(dist_type="gaussian", dist_param=[np.log(0.9875),np.log(0.1)],batch_size=10000)


h1,h2=hist_var(current, df,252, 0.05)
n1,n2=norm_linear_var(current, df, 252, 0.05)
# mc=monte_carlo_var(current.T, df, 252, 0.05)#todo fix


