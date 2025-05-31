# Image Authenticity Classification
Final Project for the Neural Networks course 2025/1 at UEA. 

This project focuses on binary image classification to distinguish between real and AI-generated synthetic images using convolutional neural networks (CNNs).

---

## 📦 Dataset

**CIFAKE – Real and AI-Generated Synthetic Images**
J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images," in IEEE Access, vol. 12, pp. 15642-15650, 2024, doi: 10.1109/ACCESS.2024.3356122.
**Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

**Content:**
- **60,000 real images** from CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **60,000 synthetic images** generated with Stable Diffusion v1.4 to replicate CIFAR-10 categories.
- All images are RGB, 32×32 pixels.
- Split: 50,000 real + 50,000 fake for training, 10,000 real + 10,000 fake for testing.

## How to run

```bash
conda env create -f environment.yml
conda activate cifake
```
