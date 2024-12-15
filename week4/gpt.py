import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import HMC, MCMC

# Ensure reproducibility
pyro.set_rng_seed(42)


# Define the model using Pyro primitives
def model():
    # Sample x from a base distribution over [-3, 3]
    x = pyro.sample("x", dist.Uniform(-3.0, 3.0))

    # Compute the unnormalized log probability of the target distribution p(x)
    sin_x = torch.sin(x)
    cos_x = torch.cos(x)
    sin_7x = torch.sin(7 * x)

    numerator = sin_x**2 + 3 * cos_x**2 * sin_7x**2 + 1
    log_px = -0.5 * x**2 + torch.log(numerator)

    # Compute the log probability of the base distribution (Uniform)
    log_p_uniform = -torch.log(
        torch.tensor(6.0)
    )  # Uniform over [-3, 3] has log_prob = -log(6)

    # The correction factor to adjust for the difference between the base and target distributions
    log_correction = log_px - log_p_uniform

    # Use pyro.factor to incorporate the log probability correction into the model
    pyro.factor("log_correction", log_correction)

    # Return x (optional, since Pyro tracks the sample site)
    return x


if __name__ == "__main__":
    # Initialize the HMC kernel using the model
    kernel = HMC(
        model=model,
        step_size=0.1,
        num_steps=20,
        adapt_step_size=True,
        target_accept_prob=0.8,
    )

    # Number of samples and burn-in (warmup) steps
    num_samples = 5000
    warmup_steps = 500

    # Run MCMC
    mcmc = MCMC(
        kernel=kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=4,
    )
    mcmc.run()

    # Extract samples
    samples = mcmc.get_samples()["x"].numpy()

    # Estimate E[X^2]
    E_X2 = np.mean(samples**2)

    print(f"Estimated E[X^2]: {E_X2:.5f}")

    # # Plot the samples (optional)
    # plt.hist(samples, bins=50, density=True, alpha=0.7, color="blue")
    # plt.title("Histogram of Samples")
    # plt.xlabel("x")
    # plt.ylabel("Density")
    # plt.show()
