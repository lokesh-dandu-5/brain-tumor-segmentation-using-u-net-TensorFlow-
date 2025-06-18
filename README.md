# Brain Tumor Segmentation using U-Net (TensorFlow)

This project implements a deep learning-based image segmentation approach to identify brain tumors from MRI images using the U-Net architecture. The model was trained, evaluated, and visualized with real-world medical imaging data.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Problem Statement](#problem-statement)  
3. [Dataset Overview](#dataset-overview)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Model Architecture - U-Net](#model-architecture---u-net)  
6. [Loss Function & Metrics](#loss-function--metrics)  
7. [Training Setup](#training-setup)  
8. [Results and Evaluation](#results-and-evaluation)  
9. [Conclusion](#conclusion)  
10. [Links](#links)

---

##  Introduction

Brain tumor segmentation is a critical task in medical image analysis. Manual annotation is time-consuming and prone to human error. Deep learning models like U-Net have proven to be effective in automating this segmentation process.

---

##  Problem Statement

The goal is to automatically segment tumor regions in brain MRI scans using a convolutional neural network. The model should accurately distinguish tumor areas from healthy tissue in grayscale images.

---

## Dataset Overview

- The dataset contains 3024 grayscale MRI images and corresponding segmentation masks.  
- Each image is 256x256 in resolution.  
- Dataset used: [Brain Tumor Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

---

##  Data Preprocessing

- **Libraries Imported**: TensorFlow, NumPy, Matplotlib, etc.  
- **Images and Masks Loaded**: All images were read and stored in lists.  
- **Converted to NumPy Arrays**: For model training and processing.  
- **Normalization**: Pixel values scaled between 0 and 1.  
- **Binarization**: Masks converted to binary for segmentation.  
- **Train-Test Split**: Dataset split into training and validation sets (80:20).  
- **Visualization**: Sample image and corresponding mask displayed for verification.

---

##  Model Architecture - U-Net

A custom U-Net model was implemented with the following encoder-decoder architecture:

- **Encoder**: 16 → 32 → 64 → 128 → 256  
- **Bottleneck**: 512  
- **Decoder**: 256 → 128 → 64 → 32 → 16  
- Each block includes:  
  - `Conv2D` layers with ReLU  
  - `Dropout` layers for regularization  
  - `MaxPooling2D` and `Conv2DTranspose` for downsampling/upsampling  
  - `Skip Connections` between encoder and decoder

---

## Loss Function & Metrics

- **Loss**: `Dice Loss` — suitable for segmentation tasks.  
- **Metric**: `Dice Coefficient` — measures overlap between predicted and actual masks.  
- Custom loss and metric functions were defined to suit the binary segmentation objective.

---

##  Training Setup

- **Epochs**: 100  
- **Learning Rate**: 1e-4 with Adam optimizer

---

## Results and Evaluation

- **Final Dice Coefficient**:  
  - Training: ~0.9280  
  - Validation: ~0.7591  
- **Loss**:  
  - Training: ~0.072  
  - Validation: ~0.2375  
- **Evaluation**:  
  - Plotted Dice coefficient and loss curves  
  - Displayed predictions on test images alongside original and ground truth masks

---

## Conclusion

This project successfully demonstrates how a U-Net model can be trained to segment brain tumors from MRI images with decent accuracy. With further tuning and data enhancements, it can be improved and potentially used in real clinical decision support systems.

---

##  Links

- 📊 **Kaggle Notebook**: [https://www.kaggle.com/code/lokeshdandu/brain-tumor-segmentation-using-u-net-tensorflow]  
- 💼 **LinkedIn Profile**: [https://www.linkedin.com/in/lokesh-dandu-583a7636a/]  
- 📖 **Medium Blog**: [https://medium.com/@lokeshdandu5/brain-tumor-segmentation-using-u-net-tensorflow-a-deep-learning-project-060541be52f5]

---

> If you find this project useful, please ⭐ the repo, share the blog, or connect with me!

