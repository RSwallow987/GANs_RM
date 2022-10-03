#VaR Backtesting Final Models
from vanilla_gam import GNet,Generator2, Generator_z2
from utils import data_sampler2, gen_kde, image_name, moments_test, mixtureofnormals
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# data_set = data_sampler2("gaussian", (0.,0.02), (252,1))
data_set=mixtureofnormals((1,0.2),(2,0.2),(0.5,0.5),252,(252,1))
z=20
gen = Generator_z2(z_dim=z)
# gen=Generator2()


gen.load_state_dict(torch.load(f='../checkpoints/MMD_final_02-10-2022-12-27-27.pt', map_location='cpu'))

#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

noise = data_sampler2(noise_dist, noise_param, (100000,z))
transformed_noise = gen.forward(noise)
transformed_noise = transformed_noise.data.numpy().reshape(100000)

x1,x2 =gen_kde(transformed_noise)
plt.savefig(image_name("MMD"))
plt.show()

#Backtest
var95=np.quantile(transformed_noise,0.05)
var99=np.quantile(transformed_noise,0.01)

k=data_set.reshape(-1).detach().numpy()
breeches=np.where(k<var95)
num_breeches=len(breeches[0])
breeches99=np.where(k<var99)

if num_breeches>len(k)*0.05:
    print("GAN: Out of Sample Breaches 95%:",num_breeches*100/len(k))
    print("GAN: Out of Sample Breaches 99%:", len(breeches99[0]) * 100 / len(k))
else:
    print("GAN: Adequate Model: Out of Sample Breeches 95%:",num_breeches*100/len(k))
    print("GAN: Adequate Model: Out of Sample Breeches 99%:", len(breeches99[0]) * 100 / len(k))

#Backtest in sample
x=torch.load(f='../data quantiles/MMD_02-10-2022-12-27-27.pt')
x=x.reshape(-1).detach().numpy()
breeches_insample=np.where(x<var95)
breeches_insample99=np.where(x<var99)
num_breeches_in_sample=len(breeches_insample[0])

if num_breeches_in_sample>len(x)*0.05:
    print("GAN: In Sample Breaches 95%:",num_breeches_in_sample*100/len(x))
    print("GAN: Out of Sample Breaches 99%:", len(breeches_insample99[0]) * 100 / len(x))
else:
    print("GAN: Adequate Model: In Sample Breeches 95%:",num_breeches_in_sample*100/len(x))
    print("GAN: Adequate Model: In Sample Breeches 99%:",  len(breeches_insample99[0]) * 100 / len(x))

#Backtest Historical Model
mu=x.mean()
sig=x.std()
var95_hist=np.quantile(x,0.05)
var99_hist=np.quantile(x,0.01)
breeches_hist=np.where(k<var95_hist)
breeches_hist99=np.where(k<var99_hist)
num_breeches_hist=len(breeches_hist[0])

if num_breeches_hist>len(x)*0.05:
    print("Breaches (Hist) 95%:",num_breeches_hist*100/len(k))
    print("Breaches (Hist) 99%:", len(breeches_hist99[0]) * 100 / len(k))
else:
    print("Adequate Model: (Hist) 95%:",num_breeches_hist*100/len(k))
    print("Adequate Model: (Hist) 99%:",len(breeches_hist99[0]) * 100 / len(k))

#Visualize
df=pd.DataFrame()

df['Actual']=pd.Series(x.flatten())
df['Generated']=pd.Series(transformed_noise.flatten())
fig=sns.kdeplot(df['Actual'], shade=True, color='r')
fig=sns.kdeplot(df['Generated'], shade=True, color='b')
plt.legend(labels=["Actual","Generated"])
plt.xlabel("")
plt.show()


#Stats Testing
ks_test=stats.ks_2samp(x, transformed_noise,alternative='two-sided')
if ks_test.pvalue <0.05:
    print("p-value is lower than our threshold of 0.05, so we reject the null hypothesis in favor of the default “two-sided” alternative: the data were not drawn from the same distribution. P-value: ", ks_test.pvalue)
else:
    print("Null Hypothesis accepted: From same distributionFrom same distribution. P-value: ", ks_test.pvalue)

print("Wasserstein Distance: ", stats.wasserstein_distance(x,transformed_noise))
print("Real Data: Mean",stats.describe(x))
print("Generated Data: ", stats.describe(transformed_noise))

x1,x2=gen_kde(transformed_noise.reshape(-1))

real_moments=stats.describe(x)
real_moments2=stats.describe(data_sampler2("gaussian", (0.,0.02), (10000,1)).detach().numpy())
generated_moments=stats.describe(transformed_noise)

# mu=[]
# var=[]
# sk=[]
# kur=[]
# mug=[]
# varg=[]
# skg=[]
# kurg=[]
# for i in range(0,1000):
#     noise = data_sampler2(noise_dist, noise_param, (100000, z))
#     transformed_noise = gen.forward(noise)
#     transformed_noise = transformed_noise.data.numpy().reshape(100000)
#     generated_moments = stats.describe(transformed_noise)
#     x1,x2,x3,x4=moments_test(real_moments,generated_moments)
#     mu.append(x1)
#     var.append(x2)
#     sk.append(x3)
#     kur.append(x4)
#     x1, x2, x3, x4 = moments_test(real_moments2, generated_moments)
#     mug.append(x1)
#     varg.append(x2)
#     skg.append(x3)
#     kurg.append(x4)
#
# print("Results for historical : mu=",sum(mu)/len(mu)," var=", sum(var)/len(var)," skew=", sum(sk)/len(sk), " kurtosis=", sum(kur)/len(kur))
# print("Results for 10000 set : mu=",sum(mug)/len(mug)," var=", sum(varg)/len(varg)," skew=", sum(skg)/len(skg), " kurtosis=", sum(kurg)/len(kurg))


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
gmm.fit(data_set.detach().numpy())
print("Log Likelihood:", gmm.score(transformed_noise.reshape(-1, 1))) #Compute the per-sample average log-likelihood of the given data X.
print("Means - Mixture:", gmm.means_)
print("Co-variances - Mixture:", gmm.covariances_)
