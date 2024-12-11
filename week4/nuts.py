import arviz as az
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as pdist
import torch
from torch.distributions import constraints


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


if __name__ == "__main__":
    # Run HMC / NUTS
    # Initial parameters for each chain
    initial_params = {
        "x": torch.tensor([-2.0, 2.0, -1.0, 1.0])  # Shape [num_chains]
    }

    nuts_kernel = pyro.infer.NUTS(model, jit_compile=True)

    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=1000,
        num_chains=4,
        warmup_steps=500,
        initial_params=initial_params,
    )

    mcmc.run()

    # Get posterior samples
    posterior_samples = mcmc.get_samples()["x"]

    # Calculate E[x^2]
    E_x2 = torch.mean(posterior_samples**2).item()
    print(f"Estimated E[X^2] using HMC/NUTS: {E_x2:.4f}")

    # Arviz
    ## Convert to Arviz InferenceData
    data = az.from_pyro(mcmc)

    # ESS, r-hat
    summary = az.summary(data)
    print(summary)

    # Density plot
    az.plot_posterior(data)
    plt.show()

    # Trace plot
    az.plot_trace(data, kind="rank_bars")
    plt.show()
