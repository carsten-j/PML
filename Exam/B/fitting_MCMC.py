import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch

# pyro.set_rng_seed(98728292)


def g(x):
    return -(torch.sin(6 * np.pi * x) ** 2) + 6 * x**2 - 5 * x**4 + 3 / 2


def gen_data():
    l = 30
    x = (torch.arange(1, l + 1) - 1) / (l - 1)
    y = g(x) + np.sqrt(0.01) * torch.randn(len(x))
    permutation = torch.randperm(len(x))
    x_shuffle = x[permutation]
    y_shuffle = y[permutation]
    x_train = x_shuffle[:20]
    y_train = y_shuffle[:20]
    x_eval = x_shuffle[20:]
    y_eval = y_shuffle[20:]
    return x_train, y_train, x_eval, y_eval


def log_likelihood(x, y, params):
    rbf = gp.kernels.RBF(input_dim=1, variance=params[1], lengthscale=params[0])
    periodic = gp.kernels.Periodic(
        input_dim=1, period=params[3], lengthscale=params[2], variance=params[4]
    )
    kernel = gp.kernels.Sum(kern0=rbf, kern1=periodic)
    noise_y = params[5]
    n = len(y)
    K = kernel.forward(x)
    like1 = -1 / 2 * y @ torch.linalg.inv(noise_y * torch.eye(n) + K) @ y
    like2 = -1 / 2 * torch.log(torch.linalg.det(noise_y * torch.eye(n) + K))
    like3 = -n / 2 * torch.log(2 * torch.tensor(np.pi))
    return like1 + like2 + like3


def approximate_log_likelihood(x, y, posterior_samples):
    log_likelihoods = []
    for rbf_l, rbf_v, per_l, per_p, per_v, noise in zip(
        posterior_samples["kernel.kern0.lengthscale"],
        posterior_samples["kernel.kern0.variance"],
        posterior_samples["kernel.kern1.lengthscale"],
        posterior_samples["kernel.kern1.period"],
        posterior_samples["kernel.kern1.variance"],
        posterior_samples["noise"],
    ):
        # print(rbf_l, rbf_v, per_l, per_p, per_v, noise)
        log_likelihoods.append(
            log_likelihood(x, y, [rbf_l, rbf_v, per_l, per_p, per_v, noise])
        )
    return torch.tensor(log_likelihoods).mean()


# Plot of data generating function
xs = torch.linspace(0, 1.0, 500)
ys = g(xs)
plt.plot(xs, ys)


x_train, y_train, x_test, y_test = gen_data()

# Defining our kernels and GP-model
rbf = gp.kernels.RBF(
    input_dim=1, variance=torch.tensor(1.0), lengthscale=torch.tensor(0.9)
)
periodic = gp.kernels.Periodic(
    input_dim=1,
    period=torch.tensor(0.5),
    lengthscale=torch.tensor(1.0),
    variance=torch.tensor(1.0),
)
kernel = gp.kernels.Sum(kern0=rbf, kern1=periodic)
gpr = gp.models.GPRegression(x_train, y_train, kernel=kernel, noise=torch.tensor(0.01))

# Putting priors on our kernel parameters
gpr.kernel.kern0.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
gpr.kernel.kern0.variance = pyro.nn.PyroSample(dist.LogNormal(1.0, 1.0))
gpr.kernel.kern1.period = pyro.nn.PyroSample(dist.Exponential(4.0))
gpr.kernel.kern1.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
gpr.kernel.kern1.variance = pyro.nn.PyroSample(dist.LogNormal(1.0, 1.0))
gpr.noise = pyro.nn.PyroSample(dist.Gamma(1.0, 1.0))


nuts_kernel = pyro.infer.NUTS(gpr.model, jit_compile=True)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=500, num_chains=2, warmup_steps=1000)
mcmc.run()


arviz_data = az.from_pyro(mcmc)
az.plot_trace(arviz_data)
plt.tight_layout()
plt.savefig("arviz_trace.png", dpi=400)
az.plot_posterior(arviz_data)
plt.tight_layout()
plt.savefig("arviz_posterior.png", dpi=400)
summ = az.summary(arviz_data)


summ = az.summary(arviz_data)
df = pd.DataFrame(summ)
print(df.to_latex())


test_loglikelihoods = []
generated_data_list = []

for i in range(20):
    print("Beginning iteration", i)
    pyro.clear_param_store()
    x_train, y_train, x_test, y_test = gen_data()
    generated_data_list.append((x_train, y_train, x_test, y_test))

    # Defining our kernels and GP-model
    rbf = gp.kernels.RBF(
        input_dim=1, variance=torch.tensor(1.0), lengthscale=torch.tensor(0.9)
    )
    periodic = gp.kernels.Periodic(
        input_dim=1,
        period=torch.tensor(0.5),
        lengthscale=torch.tensor(1.0),
        variance=torch.tensor(1.0),
    )
    kernel = gp.kernels.Sum(kern0=rbf, kern1=periodic)
    gpr = gp.models.GPRegression(
        x_train, y_train, kernel=kernel, noise=torch.tensor(0.01)
    )

    # Putting priors on our kernel parameters
    gpr.kernel.kern0.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
    gpr.kernel.kern0.variance = pyro.nn.PyroSample(dist.LogNormal(1.0, 1.0))
    gpr.kernel.kern1.period = pyro.nn.PyroSample(dist.Exponential(4.0))
    gpr.kernel.kern1.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
    gpr.kernel.kern1.variance = pyro.nn.PyroSample(dist.LogNormal(1.0, 1.0))
    gpr.noise = pyro.nn.PyroSample(dist.Gamma(1.0, 1.0))

    # SVI with delta distribution as guide

    nuts_kernel = pyro.infer.NUTS(gpr.model, jit_compile=True)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=500, num_chains=2, warmup_steps=500)
    mcmc.run()
    samples_post = mcmc.get_samples()

    test_loglikelihoods.append(approximate_log_likelihood(x_test, y_test, samples_post))


print(list(samples_post.values())[0].shape)
print(test_loglikelihoods)


map_log_likelihood = [
    -10.591714859008789,
    -9.883598327636719,
    -10.453899383544922,
    -9.179474830627441,
    -9.62856674194336,
    -9.440887451171875,
    -5.739197254180908,
    -8.277508735656738,
    -9.285099029541016,
    -6.807969093322754,
    -7.776616096496582,
    -8.214893341064453,
    -11.072059631347656,
    -10.004533767700195,
    -8.960026741027832,
    -11.985212326049805,
    -8.077812194824219,
    -10.363739967346191,
    -10.677692413330078,
    -7.064223766326904,
]


# # mcmc_log_likelihood = [tensor.to_numpy() for tensor in test_loglikelihoods]
# fig, ax = plt.subplots(figsize=(3, 3))
# ax.errorbar(
#     1,
#     np.mean(map_log_likelihood),
#     yerr=np.std(map_log_likelihood, ddof=1),
#     ecolor="k",
#     capsize=3,
#     elinewidth=1,
#     color="b",
#     fmt="o",
# )
# ax.errorbar(
#     2,
#     np.mean(test_loglikelihoods),
#     yerr=np.std(test_loglikelihoods, ddof=1),
#     ecolor="k",
#     capsize=3,
#     elinewidth=1,
#     color="r",
#     fmt="o",
# )
# ax.set_xticks([1, 2], ["MAP", "NUTS"])
# ax.set_xlim(0, 3)
# ax.set(ylabel="Posterior log likelihood")
# plt.tight_layout()
# plt.savefig("B1_loglikelihood_comp.png", dpi=400)
# plt.show()
# print("MAP:", np.mean(map_log_likelihood), np.std(map_log_likelihood, ddof=1))
# print("NUTS:", np.mean(test_loglikelihoods), np.std(test_loglikelihoods, ddof=1))
