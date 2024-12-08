# %%
import pyro
import pyro.distributions as pdist
import torch

# import arviz
# %%
from torch.distributions import constraints


class MyDensity(pdist.TorchDistribution):
    # The integration interval
    support = constraints.interval(torch.tensor(-3), torch.tensor(3))
    arg_constraints = {"start": support}

    def __init__(self, start=torch.tensor(0.0)):
        self.start = start
        super(pdist.TorchDistribution, self).__init__()

    def sample(self, sample_shape=torch.Size()):
        return self.start

    def log_prob(self, x):
        # Return log of the (unnormalized) density
        # return torch.exp(-x**2/2) * ((torch.sin(x))**2 + 3 * (torch.cos(x))**2 * (torch.sin(7 * x))**2 + 1)
        return -(x**2) / 2 + torch.log(
            (
                (torch.sin(x)) ** 2
                + 3 * (torch.cos(x)) ** 2 * (torch.sin(7 * x)) ** 2
                + 1
            )
        )


# %%
# Specify the model, which in our case is just our MyDensity distribution
def model():
    x = pyro.sample("x", MyDensity())


if __name__ == "__main__":
    # %%
    # Run HMC / NUTS
    # Run HMC / NUTS
    # Initial parameters for each chain
    initial_params = {
        "x": torch.tensor([-2.0, 2.0, -1.0, 1.0])  # Shape [num_chains]
    }

    nuts_kernel = pyro.infer.NUTS(model)
    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=5000,
        num_chains=4,
        warmup_steps=500,
        # initial_params=initial_params,
    )
    mcmc.run()

    # %%
    # Get the samples
    samples = mcmc.get_samples()["x"]

    # Calculate E(x^2)
    print(torch.mean(samples**2).item())

    # %%
    # TODO use arviz to summarize and investigate: plot
    # data = arviz.from_pyro(mcmc)

    # summary = None

    # # %%
    # # HMC / NUTS
    # def HMC(n_samples):
    #     # Run HMC / NUTS
    #     nuts_kernel = None
    #     mcmc = None
    #     # Do something with the mcmc object here ...

    #     # TODO Get the samples
    #     samples = None

    #     # TODO: Calculate E(x^2)
    #     expected_val = None
    #     return expected_val

    # # %%
    # # TODO make figures w.r.t. different sample sizes, samplers, etc.
