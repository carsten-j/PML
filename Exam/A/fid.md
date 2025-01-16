def calculate_fid(model, dataloader, device, num_samples=None):
    """Calculate FID score between real and generated images"""
    fid = FrechetInceptionDistance(normalize=True).to(device)

    if not num_samples:
        num_samples = 10000
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
    ])
    
    # Generate fake images
    model.eval()
    with torch.no_grad():
        for real_images, _ in dataloader:
            batch_size = real_images.shape[0]
            if isinstance(model, SDEDiffusion):
                samples = model.sample((batch_size, 28 * 28))
            else:
                samples = model.sample((batch_size, 28 * 28))
                samples = samples.view(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
            samples = (samples + 1) / 2 # Map from [-1, 1] to [0, 1]
            samples = samples.repeat(1, 3, 1, 1)  
            samples = transform(samples)  # Resize to (N, 3, 299, 299)
            fid.update(samples.to(device), real=False)
            if fid.real_features_num_samples >= num_samples:
                break
            # Get real images
            real_images = real_images.view(-1, 1, 28, 28)  
            real_images = (real_images + 1) / 2  
            real_images = real_images.repeat(1, 3, 1, 1)  # Repeat channels to get (N, 3, 28, 28)
            real_images = transform(real_images) # Resize to (N, 3, 299, 299)
            fid.update(real_images.to(device), real=True)

    
    fid_score = fid.compute()

    return float(fid_score)

--

def calculate_fid(model, dataloader, device, num_samples=10000):
    """
    Calculate the Fréchet Inception Distance (FID) score between real and generated images.
    
    Parameters:
        model: The generative model used to produce images.
        dataloader: DataLoader providing batches of real images.
        device: The device (CPU or GPU) to perform computations on.
        num_samples: The number of samples to use for FID calculation (default is 10,000).
    
    Returns:
        fid_score: The calculated FID score as a floating-point number.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torchmetrics.image.fid import FrechetInceptionDistance

    # Initialize FID metric
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    # Transformation to resize images to 299x299 (required for Inception model)
    transform = transforms.Resize((299, 299))

    # Set the model to evaluation mode
    model.eval()

    # Keep track of the number of processed samples
    total_fake_samples = 0
    total_real_samples = 0

    with torch.no_grad():
        # Generate and collect fake images
        while total_fake_samples < num_samples:
            batch_size = min(dataloader.batch_size, num_samples - total_fake_samples)

            # Generate fake images
            if isinstance(model, SDEDiffusion):
                fake_images = model.sample((batch_size, 28 * 28))
            else:
                fake_images = model.sample((batch_size, 28 * 28))
                fake_images = fake_images.view(-1, 1, 28, 28)

            # Process fake images
            fake_images = (fake_images + 1) / 2  # Map from [-1, 1] to [0, 1]
            fake_images = fake_images.repeat(1, 3, 1, 1)  # Convert to 3 channels
            fake_images = transform(fake_images)  # Resize to (N, 3, 299, 299)

            # Update FID metric with fake images
            fid_metric.update(fake_images.to(device), real=False)
            total_fake_samples += batch_size

        # Collect real images
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)

            # Only process up to num_samples
            if total_real_samples + batch_size > num_samples:
                batch_size = num_samples - total_real_samples
                real_images = real_images[:batch_size]

            # Process real images
            real_images = real_images.view(-1, 1, 28, 28)
            real_images = (real_images + 1) / 2  # Map from [-1, 1] to [0, 1]
            real_images = real_images.repeat(1, 3, 1, 1)  # Convert to 3 channels
            real_images = transform(real_images)  # Resize to (N, 3, 299, 299)

            # Update FID metric with real images
            fid_metric.update(real_images.to(device), real=True)
            total_real_samples += batch_size

            if total_real_samples >= num_samples:
                break

    # Compute FID score
    fid_score = fid_metric.compute()
    return float(fid_score)

--

Certainly! Computing the Fréchet Inception Distance (FID) for the MNIST dataset involves some unique considerations due to the nature of the dataset and the characteristics of the Inception network used in the computation. 

In this response, I'll:

1. **Discuss the applicability of FID to the MNIST dataset.**
2. **Outline how to compute FID using `FrechetInceptionDistance` from `torchmetrics.image.fid` for MNIST.**
3. **Explain the complications and possible solutions when applying FID to MNIST.**

---

### **1. Applicability of FID to the MNIST Dataset**

#### **Overview of MNIST**

- **MNIST Dataset Characteristics:**
  - Contains **grayscale** images.
  - Image size is **28x28 pixels**.
  - Comprises handwritten digits (0-9).
  - Single-channel images (1 channel).

#### **Challenges with FID on MNIST**

- **Mismatch with Inception Network Requirements:**
  - The Inception v3 network used in FID computation is **pre-trained on ImageNet**, which consists of **RGB images** with higher resolutions.
  - Inception expects **3-channel (RGB)** images, typically of size **299x299 pixels**.

- **Statistical Distribution Differences:**
  - The feature representations extracted from MNIST images using Inception v3 might not be meaningful due to the domain difference.
  - This can lead to inaccurate FID scores that do not reflect the true similarity between real and generated MNIST images.

---

### **2. Computing FID Using `FrechetInceptionDistance` from `torchmetrics` for MNIST**

#### **Step-by-Step Outline**

**Step 1: Setup and Imports**

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

**Step 2: Prepare the MNIST Dataset**

- **Define Transformations:**
  - Convert images to RGB by repeating the single channel 3 times.
  - Resize images to the size expected by Inception v3 (e.g., 299x299).
  - Normalize images to the range expected by Inception v3.

```python
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to 299x299 pixels
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet std deviations
])
```

- **Load Real and Generated Datasets:**

Assuming you have a dataset of generated MNIST images:

```python
# Real MNIST data
real_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)

# Generated MNIST data
# Replace 'GeneratedMNISTDataset' with your own dataset class or method to load generated images
generated_dataset = GeneratedMNISTDataset(transform=transform)
generated_loader = DataLoader(generated_dataset, batch_size=64, shuffle=False)
```

**Step 3: Initialize Frechet Inception Distance Metric**

```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize FID metric
fid_metric = FrechetInceptionDistance().to(device)
```

**Step 4: Update the Metric with Real and Generated Images**

```python
# Iterate over real images
for images, _ in real_loader:
    images = images.to(device)
    fid_metric.update(images, real=True)

# Iterate over generated images
for images in generated_loader:
    images = images.to(device)
    fid_metric.update(images, real=False)
```

**Step 5: Compute the FID Score**

```python
# Compute FID
fid_score = fid_metric.compute()
print(f"FID Score: {fid_score.item()}")
```

---

### **3. Complications and Solutions When Applying FID to MNIST**

#### **Complications**

1. **Image Dimension and Channel Mismatch:**

   - **Problem:** Inception v3 expects 3-channel RGB images of size 299x299 pixels, while MNIST images are single-channel grayscale images of size 28x28 pixels.
   - **Implication:** Directly feeding MNIST images into Inception v3 without proper preprocessing will result in errors or meaningless activations.

2. **Domain Gap Between Datasets:**

   - **Problem:** Inception v3 is trained on ImageNet, which consists of natural images vastly different from handwritten digits in MNIST.
   - **Implication:** The high-level features extracted by Inception may not be appropriate for MNIST, leading to unreliable FID scores.

3. **Statistical Representation Issues:**

   - **Problem:** The feature space used in FID computation may not capture the nuances of MNIST images.
   - **Implication:** The computed means and covariances may not accurately represent the distributions of the real and generated MNIST images.

#### **Possible Solutions**

1. **Preprocessing Images to Match Inception v3 Input Requirements:**

   - **Convert Grayscale to RGB:**
     - Duplicate the single channel to create a 3-channel image.
     - Use `transforms.Grayscale(num_output_channels=3)`.

   - **Resize Images:**
     - Scale up images to 299x299 pixels.
     - Use `transforms.Resize((299, 299))`.
   
   - **Normalize Images:**
     - Apply normalization using ImageNet mean and standard deviation.

   **Limitations:**
   - While this allows the images to be processed by Inception v3, the upscaled and colorized images may still not produce meaningful activations due to the content mismatch.

2. **Using a Feature Extractor More Suitable for MNIST:**

   - **Train a Custom Feature Extractor:**
     - Train a convolutional neural network (CNN) on MNIST classification.
     - Extract features from an intermediate layer.

   - **Use Pre-trained Models on MNIST-like Data:**
     - Use models pre-trained on datasets similar to MNIST, such as EMNIST.

   - **Implementation with TorchMetrics:**
     - Modify `FrechetInceptionDistance` to accept a custom feature extractor.

   **Example:**

   ```python
   from torchmetrics.image.fid import FrechetInceptionDistance
   from torchvision.models import resnet18

   # Define a custom feature extractor
   class MNISTFeatureExtractor(torch.nn.Module):
       def __init__(self):
           super(MNISTFeatureExtractor, self).__init__()
           self.model = resnet18(pretrained=False, num_classes=10)
           # Modify the first layer to accept 1-channel images
           self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

       def forward(self, x):
           # Extract features from an intermediate layer
           x = self.model.conv1(x)
           x = self.model.bn1(x)
           x = self.model.relu(x)
           x = self.model.maxpool(x)
           x = self.model.layer1(x)
           # Global average pooling and flatten
           x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
           x = torch.flatten(x, 1)
           return x

   # Initialize custom feature extractor
   feature_extractor = MNISTFeatureExtractor().to(device)

   # Initialize FID metric with custom feature extractor
   fid_metric = FrechetInceptionDistance(feature=64, reset_real_features=True).to(device)
   fid_metric.set_extractor(feature_extractor)
   ```

   **Note:**
   - The `torchmetrics` library may not support custom feature extractors out of the box. You may need to adjust the implementation accordingly or compute FID manually using the custom feature extractor.

3. **Adjusting FID Computation Manually:**

   - **Compute Activations Manually:**
     - Pass real and generated images through the custom feature extractor.
     - Collect features and compute means and covariances.

   - **Calculate FID Score Manually:**
     - Use the FID formula with the computed statistics.

   **Example:**

   ```python
   # Collect real features
   real_features = []
   for images, _ in real_loader:
       images = images.to(device)
       features = feature_extractor(images)
       real_features.append(features.cpu().detach())

   real_features = torch.cat(real_features, dim=0)

   # Collect generated features
   gen_features = []
   for images in generated_loader:
       images = images.to(device)
       features = feature_extractor(images)
       gen_features.append(features.cpu().detach())

   gen_features = torch.cat(gen_features, dim=0)

   # Compute mean and covariance
   mu_real = real_features.mean(dim=0)
   sigma_real = torch_cov(real_features)

   mu_gen = gen_features.mean(dim=0)
   sigma_gen = torch_cov(gen_features)

   # Compute FID
   fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
   print(f"FID Score: {fid_value}")
   ```

   **Helper Functions:**

   ```python
   def torch_cov(m):
       m = m - m.mean(dim=0)
       cov = m.t().mm(m) / (m.size(0) - 1)
       return cov

   def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
       from scipy.linalg import sqrtm

       mu1 = mu1.numpy()
       mu2 = mu2.numpy()
       sigma1 = sigma1.numpy()
       sigma2 = sigma2.numpy()

       diff = mu1 - mu2

       # Compute square root of product of covariances
       covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
       if not np.isfinite(covmean).all():
           # Add small noise to the covariances to make the computation stable
           offset = np.eye(sigma1.shape[0]) * eps
           covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

       # Check and fix imaginary parts from sqrtm
       if np.iscomplexobj(covmean):
           covmean = covmean.real

       fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
       return fid
   ```

4. **Alternatives to FID:**

   - **Use Metrics More Suitable for MNIST:**
     - **Kernel Inception Distance (KID):** Similar to FID but may be more stable for small sample sizes.
     - **Inception Score with Modified Classifier:** Use a classifier trained on MNIST.

   - **Explainability and Visualization:**
     - **t-SNE or PCA Visualization:** Visually inspect the feature distributions.
     - **Pixel-wise Metrics:** Although less suitable, sometimes MSE or SSIM can be used for initial assessments.

---

### **Summary and Recommendations**

- **Applicability of FID to MNIST:**
  - While FID is a powerful metric for evaluating generative models on datasets like ImageNet or CIFAR-10, applying it directly to MNIST requires careful handling due to the differences in data characteristics.
  
- **Computing FID with Adjustments:**
  - When using `FrechetInceptionDistance` from `torchmetrics`, preprocessing MNIST images to meet Inception v3 input requirements is necessary.
  - Consider using or training a feature extractor that is appropriate for MNIST images.

- **Complications and Solutions:**
  - The domain gap between MNIST and ImageNet can lead to unreliable FID scores.
  - Custom feature extractors can provide more meaningful feature representations for MNIST.
  - Manual computation of FID with suitable feature extractors ensures better alignment with the data.

- **Recommendations:**
  - **Use a Custom Feature Extractor:** Train or use a model appropriate for MNIST to extract features.
  - **Validate the Feature Extractor:** Ensure that the features extracted capture the important variations in MNIST digits.
  - **Complement FID with Other Metrics:** Use additional metrics and visual assessments to evaluate the quality of generated images.

---

### **Final Thoughts**

Computing FID for the MNIST dataset involves adapting the standard procedure to account for the uniqueness of the dataset. By carefully preprocessing the data and possibly customizing the feature extractor, you can obtain meaningful FID scores that help compare the performance of different generative models trained on MNIST.

Always keep in mind that metrics should be appropriate for the data and the specific characteristics of the images being analyzed. Combining quantitative metrics like FID with qualitative evaluations will give you a more comprehensive understanding of your model's performance.

If you have further questions or need assistance with specific implementation details, feel free to ask!



Computing the Fréchet Inception Distance (FID) for the MNIST dataset involves some unique considerations due to the nature of the dataset and the characteristics of the default Inception network used in the FID computation. The domain gap between MNIST and ImageNet can lead to unreliable FID scores. Here we highlight some of our concerns:
\begin{itemize}
 \item Inception expects 3-channel RGB images of size 299x299 pixels, while MNIST images are single-channel grayscale images of size 28x28 pixels.
 \item Inception is trained on ImageNet, which consists of natural images vastly different from handwritten digits in MNIST.
 The high-level features extracted by Inception may not be appropriate for MNIST,
\end{itemize}

All of the above concern can lead to inaccurate FID scores that do not reflect the true similarity between real and generated MNIST images. We also note from table , that FID when using $x_0$ to predict instead of $\epsilon$ is better which contradict the conclusion in the \cite{} paper.



- **Problem:** 
   - **Implication:** Directly feeding MNIST images into Inception v3 without proper preprocessing will result in errors or meaningless activations.

1. **Domain Gap Between Datasets:**

   - **Problem:** Inception v3 is trained on ImageNet, which consists of natural images vastly different from handwritten digits in MNIST.
   - **Implication:** The high-level features extracted by Inception may not be appropriate for MNIST, leading to unreliable FID scores.

- **Statistical Distribution Differences:**
  - The feature representations extracted from MNIST images using Inception v3 might not be meaningful due to the domain difference.
  - This can lead to inaccurate FID scores that do not reflect the true similarity between real and generated MNIST images.