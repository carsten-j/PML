import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli, Normal, kl_divergence


# Encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=4, stride=2, padding=1
        )  # (batch, 1, 28, 28) -> (batch, 32, 14, 14)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # (batch, 32, 14, 14) -> (batch, 64, 7, 7)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Bernoulli decoder
# The handwritten digits are `close’ to binary-valued, but are in fact continuous-valued.
# See histogram of pixel values in the MNIST dataset.
class BernoulliDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(BernoulliDecoder, self).__init__()
        # Fully connected layer
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        # Transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # (batch, 64, 7, 7) -> (batch, 32, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )  # (batch, 32, 14, 14) -> (batch, 1, 28, 28)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)  # Reshape
        x = F.relu(self.deconv1(x))
        x = self.deconv2(
            x
        )  # Do not use sigmoid to Output in (0,1) for Bernoulli as this is handled by the loss function
        # **Problem:** Applying the sigmoid activation separately can lead to numerical
        # instability, especially when the inputs have extreme values. Additionally,
        # it requires an extra computation step.
        return x


# The decoder for the continuous Bernoulli distribution is identical to the Bernoulli decoder
# but the loss function is different
class ContinuousBernoulliDecoder(BernoulliDecoder):
    pass


# Gaussian decoder with fixed variance
class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(GaussianDecoder, self).__init__()
        # Fully connected layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        # Deconvolutional layers to reconstruct the image
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # Upsamples from (64, 7, 7) to (32, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1
        )  # Upsamples from (32, 14, 14) to (1, 28, 28)

    def forward(self, z):
        # Fully connected layer to expand z
        x = self.fc(z)
        # Reshape into feature maps
        x = x.view(-1, 64, 7, 7)
        # Deconvolution layers with activations
        x = F.relu(self.deconv1(x))
        # Final deconvolution layer; no activation function
        x = self.deconv2(x)
        # Output is (batch_size, 1, 28, 28)
        return x  # Outputting raw values; no activation function


class GaussianDecoderLearnedVariance(nn.Module):
    def __init__(self, latent_dim):
        super(GaussianDecoderLearnedVariance, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.final_mu = nn.Conv2d(32, 1, kernel_size=1)
        self.final_logvar = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        recon_mu = self.final_mu(x)
        recon_logvar = self.final_logvar(x)
        # Apply softplus activation to ensure positive variance

        #        - **Unbounded `recon_logvar`**: The decoder outputs `recon_logvar` without any activation function, meaning it can take any real value, including large negative values.
        # - **Small Variance**: Large negative values of `recon_logvar` correspond to extremely small variances after applying the exponential function.
        # - **High PDF Values**: With very small variances, the Gaussian PDF becomes extremely peaked, leading to high values for `log_prob` when the input `x` is close to `recon_mu`.
        # - **Negative NLL**: This results in negative values for the NLL, which is unexpected since the NLL should be non-negative in practical scenarios.
        #   - **Softplus Function**: \( \text{softplus}(x) = \ln(1 + e^{x}) \), which outputs values greater than 0 for real inputs.
        recon_logvar = F.softplus(recon_logvar)
        return recon_mu, recon_logvar


# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim=2, distribution="bernoulli"):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.latent_dim = latent_dim
        self.distribution = distribution

        # Select decoder based on distribution
        if distribution == "bernoulli":
            self.decoder = BernoulliDecoder(latent_dim)
        elif distribution == "gaussian":
            self.decoder = GaussianDecoder(latent_dim)
        elif distribution == "gaussian_with_learned_variance":
            self.decoder = GaussianDecoderLearnedVariance(latent_dim)
        elif distribution == "continuous_bernoulli":
            self.decoder = ContinuousBernoulliDecoder(latent_dim)
        else:
            raise NotImplementedError(
                f"Decoder for distribution '{distribution}' is not implemented."
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random tensor with same shape as std
        return mu + eps * std  # Reparameterization trick

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.distribution == "gaussian_with_learned_variance":
            recon_mean, recon_logvar = self.decoder(z)
            return (recon_mean, recon_logvar), mu, logvar
        else:
            recon = self.decoder(z)
        return recon, mu, logvar


# Loss function with modularity for different distributions
def loss_function(recon_x, x, mu, logvar, distribution):
    """
    Compute the loss function for a VAE with different decoder distributions.
    Args:
        recon_x: reconstructed input from the decoder
        x: input data
        mu: mean of the latent distribution
        logvar: log variance of the latent distribution
        distribution: decoder distribution of the VAE
    Returns:
        loss: the loss value
    """
    batch_size = x.size(0)
    # Compute Kullback-Leibler divergence term (using torch.distributions)
    q_z = Normal(loc=mu, scale=(0.5 * logvar).exp())
    p_z = Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(logvar))
    KLD = kl_divergence(q_z, p_z).sum()
    KLD /= batch_size  # Normalize by batch size

    if distribution == "bernoulli" or distribution == "continuous_bernoulli":
        # Assert that shapes are the same
        assert recon_x.shape == x.shape

        # Flatten recon_x and x to [batch_size, 784]
        recon_x = recon_x.view(batch_size, -1)
        x = x.view(batch_size, -1)

        # Reconstruction loss (using logits and no sigmoid in decoder output)
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
        BCE /= batch_size  # Normalize by batch size

        LogC = torch.tensor(0.0)

        if distribution == "continuous_bernoulli":
            cb = ContinuousBernoulli(logits=recon_x)
            LogC = cb._cont_bern_log_norm()
            LogC = torch.sum(LogC) / batch_size  # Normalize by batch size

        return BCE + KLD + LogC

    elif distribution == "gaussian":
        # Flatten recon_x and x to [batch_size, 784]
        recon_x = recon_x.view(batch_size, -1)
        x = x.view(batch_size, -1)

        # Reconstruction loss (assuming a fixed variance, can use MSE)
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        MSE /= batch_size  # Normalize by batch size

        return MSE + KLD

    elif distribution == "gaussian_with_learned_variance":
        recon_mu, recon_logvar = recon_x
        recon_mu = recon_mu.view(batch_size, -1)
        recon_logvar = recon_logvar.view(batch_size, -1)
        x = x.view(batch_size, -1)
        # Ensure that the variance used in the `Normal` distribution doesn't become too small,
        # which can be done by clamping the standard deviation:
        recon_std = torch.exp(0.5 * recon_logvar).clamp(min=1e-3)
        recon_dist = Normal(recon_mu, recon_std)
        NLL = -recon_dist.log_prob(x)
        NLL = NLL.sum() / batch_size

        return NLL + KLD

    else:
        raise NotImplementedError(
            f"Loss function for distribution '{distribution}' is not implemented."
        )
