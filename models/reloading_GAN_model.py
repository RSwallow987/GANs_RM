from vanilla_gam import Generator_z2
from utils import get_noise, data_sampler2, save_models,  getstocks
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

gen = Generator_z2()
gen.load_state_dict(torch.load(f='../checkpoints/16-07-2022-11-54-05.pt', map_location='cpu'))

#Testing
noise_dist = "gaussian"
noise_param = (0., 1.)

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