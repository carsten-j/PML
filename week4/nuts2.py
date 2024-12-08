import pyro
import torch
from pyro.infer import MCMC, NUTS


# Define the potential energy function (negative log probability)
def potential_energy(params):
    # Extract x from the params dictionary
    x = params["x"]
    # Compute log_p(x)
    term = torch.sin(x) ** 2 + 3 * torch.cos(x) ** 2 * torch.sin(7 * x) ** 2 + 1
    term = torch.clamp(term, min=1e-10)  # Avoid log(0)
    log_px = -0.5 * x**2 + torch.log(term)
    # Set log_p(x) = -inf outside [-3,3]
    log_px = torch.where(
        (x >= -3.0) & (x <= 3.0),
        log_px,
        torch.tensor(-float("inf"), dtype=x.dtype, device=x.device),
    )
    # Return negative log probability (potential energy)
    return -log_px


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')  # Commented out; not necessary unless required
    device = "mps"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    pyro.set_rng_seed(0)

    # Initial parameters for each chain
    initial_params = {
        "x": torch.tensor([-2.0, 2.0, -1.0, 1.0])  # Shape [num_chains]
    }

    # Define the NUTS kernel with the potential energy function
    nuts_kernel = NUTS(potential_fn=potential_energy)

    # Set up MCMC with multiple chains and initial parameters
    mcmc = MCMC(
        nuts_kernel,
        num_samples=4000,
        warmup_steps=500,
        num_chains=4,
        initial_params=initial_params,
    )

    # Run the MCMC sampler
    mcmc.run()

    # Extract the samples
    samples = mcmc.get_samples()
    x_samples = samples["x"]

    # Calculate E[x^2] using the collected samples
    E_x2 = (x_samples**2).mean().item()

    # Print the expected value of x squared
    print("E[x^2] =", E_x2)
