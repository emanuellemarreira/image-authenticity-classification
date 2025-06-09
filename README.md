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

---

## Architecture

- ### ResNet50
    
    ResNet50 is a 50-layer deep convolutional neural network from the ResNet
    family (Residual Networks), widely used in image classification and feature extraction. The architecture of ResNet50 is divided into four main parts: the convolutional layers, the identity block, the convolutional block, and the fully connected layers. The convolutional layers are responsible for extracting features from the input image, while the identity block and convolutional block are responsible for processing and transforming these features. Finally, the fully connected layers are used to make the final classification.

    <img align="center" alt="ResNet50 Model Architecture" src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*VM94wVftxP7wkiKo4BjfLA.png">

    Source: [Exploring ResNet50: An In-Depth Look at the Model Architecture and Code Implementation](https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f)

- ### MobileNetV2

    MobileNetV2 is an improved version of the original MobileNet architecture, designed specifically for efficient deep learning on mobile and embedded devices. 
    t is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

---

## How to run

```bash
conda env create -f environment.yml
conda activate cifake
```

---

## Team

1. Afonso Henrique Torres Lucas | ifonso.developer@gmail.com | iFonso
2. Emanuelle Rocha Marreira | emanuellemarreira@gmail.com | emanuellemarreira
3. Erik Gustavo Lima de Oliveira | erik.exatas10@gmail.com | ErikExatas
4. David Augusto De Oliveira E Silva | david-augusto-silva
5. Ítalo Ferreira Fonseca | ItaloFonseca
6. João Vitor Silva De Carvalho | joaov1524@gmail.com | joaocarvalhov
7. Lilian Iazzai De Souza Oliveira | lilianiazzai@gmail.com | lilianiazzai
8. Vitor Nascimento Aguiar | Vtaguiar1909 

---

This project was developed under the guidance of our professor Elloá B. Guedes (ebgcosta@uea.edu.br) in the course Redes Neurais Artificiais 2025.1