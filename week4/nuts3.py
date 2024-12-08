import pyro
import torch
from pyro.distributions import TorchDistribution
from pyro.infer.mcmc import MCMC, NUTS
from torch.distributions import constraints


class CustomDistribution(TorchDistribution):
    support = constraints.real
    has_rsample = False  # No reparameterized sample method

    def __init__(self, validate_args=None):
        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size(),
            validate_args=validate_args,
        )

    def log_prob(self, value):
        # Compute the unnormalized log probability
        log_prob = -0.5 * value**2 + 2 * torch.log(torch.sin(2 * value).abs() + 1e-8)
        return log_prob

    # Since we cannot sample from this distribution directly, we can omit sample methods
    def sample(self, sample_shape=torch.Size()):
        return self.start

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError(
            "Reparameterized sampling is not implemented for CustomDistribution."
        )


def model():
    x = pyro.sample("x", CustomDistribution())
    # You can define observations or further computations involving 'x'
    # For example, let's assume we have observed data 'y' that's normally distributed around 'x'
    # y = pyro.sample("y", dist.Normal(x, 1.0), obs=torch.tensor(0.5))


# Define the NUTS kernel
nuts_kernel = NUTS(model)

# Run MCMC
mcmc_run = MCMC(nuts_kernel, num_samples=20000, warmup_steps=2000)
mcmc_run.run()


### Extract Samples


# Get samples from the MCMC run
samples = mcmc_run.get_samples()
x_samples = samples["x"]

E_x2 = (x_samples**2).mean().item()

# Print the expected value of x squared
print("E[x^2] =", E_x2)
# print("Sampled x values:")
# print(x_samples)
