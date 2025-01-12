from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

# This UNET-style prediction model was originally included as part of the Score-based generative modelling tutorial
# by Yang Song et al: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        marginal_prob_std: Callable[[torch.Tensor], torch.Tensor] | None,
        channels=(32, 64, 128, 256),
        embed_dim=256,
    ):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        # Incorporate information from t
        h1 += self.dense1(embed)
        # Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        # Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output (if given a std function)
        if self.marginal_prob_std is not None:
            h /= self.marginal_prob_std(t)[:, None, None, None]
        return h


# ExponentialMovingAverage implementation as used in pytorch vision
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, avg_model, decay, avg_device="cpu"):
        def ema_avg(avg_model_param, model_param, _num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        if isinstance(avg_device, str):
            avg_device = torch.device(avg_device)

        super().__init__(avg_model, avg_device, ema_avg, use_buffers=True)


class DDPM(nn.Module):
    def __init__(
        self,
        network,
        max_t: int = 100,
        beta_1: float = 1e-4,
        beta_max_t: float = 2e-2,
        predict_mean_by: Literal["e", "u", "x0"] = "x0",
        reduce_variance_by: Literal[
            "low-discrepency", "importance-sampling", "importance-batch", None
        ] = None,
    ):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_max_t: [float]
            beta_t value at t=T (last step)
        max_t: int
            The number of diffusion steps.
        predict_mean_by: str
            Which target (epsilon, x_t-1 mean, x_0 mean) the network should predict.
        reduce_variance_by: str | None
            An optional method to reduce the variance of the loss estimate.
        """

        super().__init__()

        # Normalize time input before evaluating neural network
        # Reshape input into image format and normalize time value before sending it to network model
        self._network = network

        # Total number of time steps
        self.max_t = max_t
        self.predict_mean_by = predict_mean_by
        self.reduce_variance_by = reduce_variance_by

        # Registering as buffers to ensure they get transferred to the GPU automatically
        self.register_buffer("beta", torch.linspace(beta_1, beta_max_t, max_t + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

        if self.reduce_variance_by in ("importance-sampling", "importance-batch"):
            self.imps_warmup = True
            history_length = 10
            self.register_buffer(
                "imps_idx", torch.zeros((self.max_t + 1,), dtype=torch.int64)
            )
            self.register_buffer(
                "imps_hist", torch.zeros((self.max_t + 1, history_length))
            )
            startup_ts = (
                torch.arange(1, self.max_t + 1)
                .expand((history_length, max_t))
                .flatten()
            )
            self.register_buffer(
                "imps_startup_ts", startup_ts[torch.randperm(startup_ts.numel())]
            )
            self.imps_startup_idx = 0

    def network(self, x, t):
        return self._network(
            x.reshape(-1, 1, 28, 28), t.squeeze() / self.max_t
        ).reshape(-1, 28 * 28)

    def forward_diffusion(self, x0, t, epsilon):
        """
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        """

        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: torch.tensor (unit)
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """
        if self.predict_mean_by == "e":
            mean = (
                1.0
                / torch.sqrt(self.alpha[t])
                * (
                    xt
                    - (self.beta[t])
                    / torch.sqrt(1 - self.alpha_bar[t])
                    * self.network(xt, t)
                )
            )
        elif self.predict_mean_by == "u":
            mean = self.network(xt, t)
        elif self.predict_mean_by == "x0":
            full_epsilon = xt - self.network(xt, t)
            mean = (1.0 / torch.sqrt(1 - self.beta[t])) * (
                xt - (self.beta[t] / torch.sqrt(1 - self.alpha_bar[t])) * full_epsilon
            )
        else:
            raise ValueError(f"Invalid value: {self.predict_mean_by=}")

        mask = t > 0
        std = torch.where(
            mask,
            torch.sqrt(
                ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
            ),
            0,
        )

        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xt = torch.randn(shape).to(self.beta.device)
        for t in range(self.max_t, 0, -1):
            noise = torch.randn_like(xt) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion(xt, t, noise)

        return xt

    def sample_ts(self, x0):
        b_size = x0.shape[0]
        if self.reduce_variance_by is None:
            t = torch.randint(1, self.max_t, (b_size, 1)).to(x0.device)
            t_weights = torch.ones_like(t)
        elif self.reduce_variance_by == "low-discrepency":
            # Remove last, since it's equal under modulus. Add extra one to compensate rounding at start and end
            ts_base = torch.linspace(1, self.max_t + 1, b_size + 1)[:-1]
            # Random addition between 0 and the step size of ts_base
            ts_offset = torch.rand_like(ts_base) * ts_base[1]
            # Taking a t value from a random index, yields a marginal distribution that is uniform over the reals [1,
            # max_t+1). Taking the floor, yields a marginal distribution that is unfirom over the integers [1,
            # max_t] Clamp to ensure floating point errors don't cause out-of-bounds Unlike "Variational Diffusion
            # Models", individual positions (e.g. the first t) are not uniformly distributed, this could be sovled
            # with a random permutation, but that is needless work.
            t = (
                (ts_base + ts_offset)
                .floor()
                .int()
                .clamp(1, self.max_t)
                .view((b_size, 1))
                .to(x0.device)
            )
            t_weights = torch.ones_like(t)
        elif self.reduce_variance_by in ("importance-sampling", "importance-batch"):
            if (
                self.imps_warmup
                and self.imps_startup_idx + b_size < self.imps_startup_ts.numel()
            ):
                t = (
                    self.imps_startup_ts[
                        self.imps_startup_idx : self.imps_startup_idx + b_size
                    ]
                    .view((b_size, 1))
                    .to(x0.device)
                )
                t_weights = torch.ones_like(t)
                self.imps_startup_idx += b_size
            elif self.imps_warmup:
                self.imps_warmup = False
                missing_warmup = (
                    self.imps_startup_idx + b_size - self.imps_startup_ts.numel()
                )
                missing_elems = torch.randint(1, self.max_t, (missing_warmup,)).to(
                    x0.device
                )
                t = torch.cat(
                    (self.imps_startup_ts[self.imps_startup_idx :], missing_elems),
                    dim=0,
                ).view((b_size, 1))
                t_weights = torch.ones_like(t)
            else:
                t = torch.randint(1, self.max_t, (b_size, 1)).to(x0.device)
                weights = torch.sqrt(self.imps_hist.mean(dim=1)).to(x0.device)
                if self.reduce_variance_by == "importance-sampling":
                    weights = (weights / weights.sum()) * b_size

                t_weights = weights[t.view(-1)].view_as(t)
        else:
            raise ValueError(f"Invalid value: {self.reduce_variance_by=}")

        if self.reduce_variance_by != "importance-sampling":
            t_weights = (t_weights / t_weights.sum()) * b_size
        return t, t_weights

    def elbo_simple(self, x0: torch.Tensor):
        """
        ELBO training objective (Algorithm 1 in Ho et al, 2020)

        Parameters
        ----------
        x0: torch.tensor
            Input image

        Returns
        -------
        float
            ELBO value
        """
        # Sample time step t and weights for each element
        t, t_weights = self.sample_ts(x0)

        # Sample noise
        epsilon = torch.randn_like(x0)

        xt = self.forward_diffusion(x0, t, epsilon)

        if self.predict_mean_by == "e":
            target = epsilon
        elif self.predict_mean_by == "u":
            target = self.forward_diffusion(x0, t - 1, epsilon)
        elif self.predict_mean_by == "x0":
            target = x0
        else:
            raise ValueError(f"Invalid value: {self.predict_mean_by=}")

        unweighted_se = nn.MSELoss(reduction="none")(target, self.network(xt, t)).mean(
            tuple(range(1, len(target.shape)))
        )

        if self.reduce_variance_by in ("importance-sampling", "importance-batch"):
            uniq_t, rev_t, count_t = torch.unique(
                t.view(-1), return_inverse=True, return_counts=True
            )
            # Nth occurence (zero-indexed) of the particular value of t
            same_t = torch.eq(rev_t.view((-1, 1)), rev_t.view((1, -1)))
            nth_occurence = torch.triu(same_t).sum(dim=0) - 1
            hist_idx = (
                self.imps_idx[t.view(-1)] + nth_occurence
            ) % self.imps_hist.shape[1]
            self.imps_hist[t.view(-1), hist_idx] = unweighted_se.detach() ** 2
            self.imps_idx[uniq_t] += count_t

        return -(unweighted_se * t_weights).mean()

    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        return -self.elbo_simple(x0)
