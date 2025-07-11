# Probabilistic Machine Learning (PML)

## University of Copenhagen - DIKU - Winter 2024

This repository contains course materials, assignments, and implementations for the Probabilistic Machine Learning course at the Department of Computer Science (DIKU), University of Copenhagen.

## 📚 Course Overview

This course covers fundamental and advanced topics in probabilistic machine learning, including Bayesian inference, variational methods, generative models, and modern deep probabilistic models. The course combines theoretical foundations with practical implementation using Python and PyTorch.

## 🗂️ Repository Structure

### Weekly Materials

- **Week 1**: Introduction to Probabilistic ML, Variational Autoencoders (VAE)
  - Basic regression and probabilistic modeling
  - VAE implementations and theory
  - Linear regression from probabilistic perspective

- **Week 2**: Expectation-Maximization and Variational Inference
  - EM algorithm implementation
  - Gaussian Mixture Models (GMM)
  - Variational inference fundamentals
  - Tutorial on GMM with solutions

- **Week 3**: Sampling Methods and Monte Carlo
  - Importance sampling
  - Numerical integration techniques
  - Cauchy distribution sampling
  - MCMC methods using Pyro

- **Week 4**: Gaussian Processes and Advanced Sampling
  - Gaussian process regression
  - NUTS (No-U-Turn Sampler) implementation
  - Stochastic Variational Inference (SVI)
  - Real-world GP applications (Mauna Loa CO₂ data)

- **Week 5**: Diffusion Models
  - Introduction to diffusion processes
  - Score-based generative models
  - Implementation of basic diffusion models

- **Week 6**: Denoising Diffusion Probabilistic Models (DDPM)
  - Advanced DDPM implementations
  - Template notebooks for DDPM training
  - Generated samples and visualizations

### Specialized Topics

- **Autoencoders/**: Various autoencoder implementations
  - Standard autoencoders
  - Variational autoencoders (VAE)
  - PCA-based autoencoders
  - Anomaly detection with autoencoders

- **CNN/**: Convolutional Neural Networks
  - MNIST digit classification with PyTorch
  - CNN architectures and best practices

- **GPs/**: Gaussian Processes
  - Scikit-learn GP implementations
  - Advanced GP modeling techniques

- **VAE/**: Advanced Variational Autoencoders
  - Complete VAE implementations
  - Training pipelines and visualizations
  - Model architectures and utilities

### Assignments and Exams

- **Assignment Solutions**: Complete solutions for course assignments
- **Exam/**: Exam materials and solutions
  - Section A: DDPM implementations and theory
  - Section B: Advanced topics and applications

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Apple Silicon Mac with MPS support (optional)

### Key Dependencies

- **PyTorch**: Deep learning framework
- **Pyro**: Probabilistic programming
- **scikit-learn**: Traditional ML algorithms
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive notebooks
- **TensorBoard**: Experiment tracking

## 🧮 Key Technologies and Frameworks

- **PyTorch**: Primary deep learning framework
- **Pyro**: Probabilistic programming and Bayesian inference
- **scikit-learn**: Gaussian processes and traditional ML
- **Arviz**: Bayesian analysis and visualization
- **TorchInfo**: Model architecture summaries

## 📝 Topics Covered

### Core Probabilistic ML Concepts

1. **Bayesian Inference**: Prior, likelihood, and posterior distributions
2. **Variational Methods**: Variational inference and approximation techniques
3. **Expectation-Maximization**: Parameter estimation for latent variable models
4. **Monte Carlo Methods**: Sampling techniques and numerical integration
5. **Gaussian Processes**: Non-parametric Bayesian modeling

### Generative Models

1. **Variational Autoencoders (VAE)**: Deep generative modeling
2. **Autoencoders**: Dimensionality reduction and representation learning
3. **Diffusion Models**: Score-based and DDPM approaches
4. **Gaussian Mixture Models**: Classical mixture modeling

### Advanced Topics

1. **Stochastic Variational Inference**: Scalable Bayesian methods
2. **MCMC Sampling**: NUTS and advanced sampling techniques
3. **Neural Networks in Bayesian Setting**: Uncertainty quantification
4. **Anomaly Detection**: Probabilistic approaches

## 🚀 Getting Started

1. **Week 1**: Start with `week1/week_1.ipynb` for basic probabilistic regression
2. **Autoencoders**: Explore `Autoencoders/Autoencoder.ipynb` for hands-on implementation
3. **VAE**: Check `week1/vae_simple.ipynb` for variational autoencoder basics
4. **Gaussian Processes**: Run `week4/gaussian_processes.ipynb` for GP introduction

## 📊 Practical Implementations

### Best Practices Demonstrated

- Reproducible research with fixed random seeds
- GPU acceleration (CUDA/MPS) when available
- Type annotations for better code quality
- Cross-validation and proper model evaluation
- Visualization of results and model behavior

## 🎯 Learning Objectives

By the end of this course, students will be able to:

1. Implement fundamental probabilistic ML algorithms from scratch
2. Apply Bayesian inference to real-world problems
3. Design and train deep generative models
4. Use modern probabilistic programming frameworks
5. Critically evaluate uncertainty in machine learning models
6. Implement advanced sampling techniques for complex distributions

## 📖 Additional Resources

- Course lecture notes available in `HTML/` directory
- Model checkpoints and trained models in respective `models/` folders
- Generated samples and experiment results in `figures/` directories
- Utility scripts for notebook management in `scripts/`

