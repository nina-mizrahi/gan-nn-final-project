# Generative Adversarial Networks (GAN) Final Project – CS152

This repository contains the final project for the Neural Networks course (CS152), where we implemented and evaluated **GAN** and **Wasserstein GAN (WGAN)** models on the MNIST dataset. We assessed the performance of the generated images using both **FID** (Fréchet Inception Distance) and **SSIM** (Structural Similarity Index) metrics.

## 🔍 Project Overview

The project demonstrates:
- Implementation of standard GAN and WGAN architectures from scratch in PyTorch
- Evaluation of generative quality using both visual inspection and quantitative metrics
- Comparison of training behavior across models using discriminator loss, generator loss, and output trends
- Saving and reloading of model parameters for long-run training checkpoints

## 🗂️ File Structure

├── GAN_params/ # Saved generator & discriminator parameters for GAN
├── WGAN_params/ # Saved generator & discriminator parameters for WGAN
├── plots/ # Plots of training losses and discriminator outputs
├── saved_loss/ # Generator and discriminator loss values (NumPy arrays)
├── GAN.py # GAN training script
├── WGAN.py # WGAN training script
├── nn_helper.py # Generator, Discriminator, and loss function classes
├── plotting.ipynb # Notebook for visualizing losses and generated outputs
├── evaluating_GAN_generator.ipynb # Evaluates GAN using FID & SSIM
├── evaluating_WGAN_generator.ipynb # Evaluates WGAN using FID & SSIM


## 📊 Evaluation Metrics

- **FID (Fréchet Inception Distance)**: Measures similarity between distributions of real and generated images using feature statistics from a pretrained Inception network.
- **SSIM (Structural Similarity Index)**: Measures image similarity based on luminance, contrast, and structure.

Both metrics are used to quantify the visual quality and realism of generated samples.

## 💻 Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-image

## 👥 Collaborators

This project was completed as a team project for CS152 (Neural Networks) at [Your College].

- Nina Mizrahi
- Kevin Phan
- James Duffy

Original group repository: [https://github.com/ph-kev/gan-nn-final-project](https://github.com/ph-kev/gan-nn-final-project)

---

## 📌 Notes

- All model checkpoints and metrics are saved and can be used for future experiments or fine-tuning.
- Plots provide insights into model convergence, discriminator behavior, and generator learning stability.
