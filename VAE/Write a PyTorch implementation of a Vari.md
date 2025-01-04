Write a PyTorch implementation of a Variational Autoencoder (VAE) for the MNIST dataset with a 2D latent space. Please ensure that the following requirements are satisfied:

- Scale and normalize the dataset
- Division of the training dataset into training and validation sets.
- A convolutional neural network (CNN) for both the encoder and decoder.
- A Gaussian distribution for q(z|x) in the encoder
- A Continuous Bernoulli distributed for p(x|z) in the decoder
- The loss function is defined as a separate method, adjusted for the distribution used for the decoder.
- Use of GPU or MPS if available.



- A summary of the model using `torchinfo`.
- Progress bars using `tqdm`.


- Visualization of the latent space, reconstructed images, and new samples from the latent space.
