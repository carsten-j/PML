import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device (MPS or GPU if available)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {device}")


save_image = True

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 2  # Reduce to 2 dimensions


# MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x > 0,
        lambda x: x.float(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# VAE model
class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Implements a Variational Autoencoder for unsupervised learning of latent representations.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layers.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        fc1 (nn.Linear): First fully connected layer of the encoder.
        fc2 (nn.Linear): Second fully connected layer of the encoder.
        fc_mu (nn.Linear): Layer that outputs the mean of the latent Gaussian distribution.
        fc_logvar (nn.Linear): Layer that outputs the log variance of the latent Gaussian distribution.
        fc3 (nn.Linear): First fully connected layer of the decoder.
        fc4 (nn.Linear): Second fully connected layer of the decoder.
        relu (nn.ReLU): ReLU activation function.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    Methods:
        encode(x):
            Encodes input data into latent space.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors of the latent distribution.

        reparameterize(mu, logvar):
            Samples from the latent space using the reparameterization trick.

            Args:
                mu (torch.Tensor): Mean tensor of the latent Gaussian distribution.
                logvar (torch.Tensor): Log variance tensor of the latent Gaussian distribution.

            Returns:
                torch.Tensor: Sampled latent vector.

        decode(z):
            Decodes latent vectors back to data space.

            Args:
                z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

            Returns:
                torch.Tensor: Reconstructed input tensor.

        forward(x):
            Performs a forward pass through the VAE.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed input, mean, and log variance tensors.
    """

    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        # Mean and log-variance for latent variables
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 512)
        self.fc5 = nn.Linear(512, 28 * 28)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """
        Encodes input data into latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors of the latent distribution.
        """
        # Encode input to latent space
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Samples from the latent space using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean tensor of the latent Gaussian distribution.
            logvar (torch.Tensor): Log variance tensor of the latent Gaussian distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Decodes latent vectors back to data space.

        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        # Decode latent variable back to image
        h3 = self.relu(self.fc3(z))
        h4 = self.relu(self.fc4(h3))
        x_recon = self.sigmoid(self.fc5(h4))
        return x_recon

    def forward(self, x):
        """
        Performs a forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed input, mean, and log variance tensors.
        """
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Loss function (combines reconstruction loss and KL divergence)
def loss_function(recon_x, x, mu, logvar):
    """
    Computes the VAE loss function as the sum of reconstruction loss and KL divergence.

    The loss function is defined as:
    Loss = Reconstruction Loss (BCE) + Kullback-Leibler Divergence (KLD)

    - **Reconstruction Loss (BCE):**
      Measures how well the decoder reconstructs the input data. It is computed using
      binary cross-entropy between the reconstructed output `recon_x` and the original input `x`.

    - **Kullback-Leibler Divergence (KLD):**
      Measures how much the learned latent distribution `q(z|x)` diverges from the prior distribution `p(z)`.
      For a VAE, it acts as a regularizer to ensure that the latent space follows a standard normal distribution.

    Args:
        recon_x (torch.Tensor): Reconstructed input of shape (batch_size, input_dim).
        x (torch.Tensor): Original input data of shape (batch_size, input_dim).
        mu (torch.Tensor): Mean of the latent Gaussian distribution (batch_size, latent_dim).
        logvar (torch.Tensor): Log variance of the latent Gaussian distribution (batch_size, latent_dim).

    Returns:
        torch.Tensor: Scalar loss value combining reconstruction loss and KL divergence.

    References:
        - Kingma, D. P., & Welling, M. (2014). **Auto-Encoding Variational Bayes**.
          *2nd International Conference on Learning Representations (ICLR)*.
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, 28 * 28), reduction="sum"
    )
    # KL divergence between the learned latent distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    return BCE + KLD


# Initialize model and optimizer
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(epoch):
    """
    Train the VAE model for one epoch.

    Args:
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.train()

    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        # Backward pass and optimization
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Epoch [{}/{}] Batch [{}/{}] Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item() / len(data),
                )
            )

    average_loss = train_loss / len(train_loader.dataset)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch + 1, average_loss))


def test(epoch):
    """
    Evaluate the model on the test dataset.

    Args:
        epoch (int): The current epoch number.

    Returns:
        None
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


for epoch in range(num_epochs):
    train(epoch)
    test(epoch)


# Visualization of the latent space
model.eval()
with torch.no_grad():
    # Determine the total number of samples in the test dataset
    total_samples = len(test_loader.dataset)

    # Preallocate arrays for latent variables and labels
    z_all = np.zeros((total_samples, latent_dim), dtype=np.float32)
    labels_all = np.zeros(total_samples, dtype=np.int64)

    # Initialize starting index
    start_idx = 0

    for data, labels in test_loader:
        batch_size = data.size(0)
        data = data.to(device)
        data = data.view(-1, 28 * 28)
        # This modification ensures that each latent variable
        # z is a sample from the learned distribution
        # ( \mathcal{N}(\mu, \sigma^2) ), providing a
        # more faithful representation of the VAE's behavior.
        mu, logvar = model.encode(data)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = z.cpu().numpy()

        end_idx = start_idx + batch_size
        z_all[start_idx:end_idx] = z
        labels_all[start_idx:end_idx] = labels

        start_idx = end_idx


# Plot the latent space
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    z_all[:, 0], z_all[:, 1], c=labels_all, cmap="tab10", alpha=0.7, s=15
)
plt.colorbar()
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent 2D Space Visualization of MNIST Digits")
plt.grid(True)
plt.tight_layout()
if save_image:
    plt.savefig("results/latent_space.png", dpi=600)
plt.show()


# Define the grid size
k = 20  # Number of points along each dimension

grid_x = torch.linspace(0.01, 0.99, k)
grid_y = torch.linspace(0.01, 0.99, k)
mesh_x, mesh_y = torch.meshgrid(
    grid_x, grid_y, indexing="xy"
)  # Ensure correct axis indexing

# Stack the grid coordinates into a shape (k*k, 2)
coords = torch.stack([mesh_x, mesh_y], dim=-1).view(-1, 2)

# Map the uniformly distributed samples to the standard normal distribution using the inverse CDF
normal = torch.distributions.normal.Normal(0, 1)
z = normal.icdf(coords)  # Shape: (k*k, 2)

# Move the model to evaluation mode
model.eval()

# Move the grid points to the same device as the model
z = z.to(device)

with torch.no_grad():
    # Decode the latent variables to generate images
    reconstructed = model.decode(z).cpu()  # Shape: (k*k, 28*28)

# Reshape the reconstructed images for visualization
images = reconstructed.view(-1, 1, 28, 28)  # Shape: (k*k, 1, 28, 28)

# Convert images to NumPy array for plotting
images = images.numpy()

# Create a figure with a grid of subplots
fig, axes = plt.subplots(k, k, figsize=(k, k))

for i in range(k):
    for j in range(k):
        idx = i * k + j  # Index of the current image
        # Reverse the order of rows to have the lowest latent y-values at the bottom
        ax = axes[k - i - 1, j]
        ax.imshow(images[idx, 0, :, :], cmap="gray")
        ax.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
if save_image:
    plt.savefig("results/generated_digits.png", dpi=600)
plt.show()
