import pickle

import arviz as az
import pandas as pd
import pyro
import pyro.distributions
import pyro.distributions as pdist
import torch
from torch.distributions import constraints

az.style.use("arviz-doc")


class CustomDensity(pdist.TorchDistribution):
    # The domain of the custom PDF
    support = constraints.interval(torch.tensor(-3), torch.tensor(3))
    # Enforce that start value is within the support
    arg_constraints = {"start": support}

    def __init__(self, start=torch.tensor(0.0)):
        self.start = start
        super().__init__()
        self._batch_shape = torch.Size()
        self._event_shape = torch.Size()

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    # This is just used as a starting point for the MCMC sampler
    def sample(self, sample_shape=torch.Size()):
        return self.start

    def log_prob(self, x):
        # Return log of the unnormalized density
        return -(x**2) / 2 + torch.log(
            (
                (torch.sin(x)) ** 2
                + 3 * (torch.cos(x)) ** 2 * (torch.sin(7 * x)) ** 2
                + 1
            )
        )


def model():
    x = pyro.sample("x", CustomDensity())


# HMC / NUTS
def HMC(n_samples):
    # num_chains = n_samples // 2
    # The is 10 cpu cores available on my machine
    if n_samples == 10:
        num_chains = 2
        no_samples = 5
    elif n_samples == 100:
        num_chains = 4
        no_samples = 25
    elif n_samples == 1000:
        num_chains = 4
        no_samples = 250
    else:
        raise ValueError("n_samples must be 10, 100 or 1000")

    nuts_kernel = pyro.infer.NUTS(model, jit_compile=True)

    tensor = torch.randn(num_chains)
    # To ensure values are within [-3, 3]
    tensor = torch.clamp(tensor, -3, 3)
    initial_params = {"x": tensor}

    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=no_samples,
        num_chains=num_chains,
        warmup_steps=500,
        initial_params=initial_params,
    )
    mcmc.run()

    samples = mcmc.get_samples()["x"]

    data = az.from_pyro(mcmc)
    rhat = az.rhat(data)
    rhat = rhat.x.values
    # print(rhat)

    expected_val = torch.mean(samples**2).item()
    return expected_val, rhat


if __name__ == "__main__":
    with open("../week3/data/data.pkl", "rb") as f:
        data = pickle.load(f)

    trials = 100
    number_of_samples = [10, 100, 1000]

    for trial in range(trials):
        for n_samples in number_of_samples:
            print(f"Trial {trial + 1}/{trials}")
            estimator, rhat = HMC(n_samples)
            # Create a string representation of q
            q_repr = "HMC/NUTS"
            data.append(
                {
                    "trial": trial,
                    "samples": n_samples,
                    "sampling method": q_repr,
                    "estimator": estimator,
                    "rhat": rhat,
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("./data/data500_4.csv", index=False)

    # Compute mean and standard deviation for each combination of n_samples and q
    grouped = df.groupby(["samples", "sampling method"])["estimator"].agg(
        ["mean", "std"]
    )

    print(grouped)
