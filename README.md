# Generative Models - CycleGAN Implementation

This repository contains implementations of CycleGAN for image-to-image translation, specifically for transforming photographs into Monet-style paintings. The project includes multiple architectural approaches and comprehensive training scripts.

## Project Structure

```
├── README.md
├── model.ipynb              # General CycleGAN implementation
├── model_unet.ipynb         # UNet-based CycleGAN implementation (Main notebook)  
└── model_old_arch.ipynb     # Legacy architecture experiments
```

## Project Overview

This project implements **CycleGAN** (Cycle-Consistent Adversarial Networks) for unpaired image-to-image translation. The goal is to transform ordinary photographs into Monet-style paintings while maintaining the ability to reconstruct the original image through cycle consistency.

### Key Features

- **Multiple Generator Architectures**: ResNet-based and UNet-based generators
- **Advanced Loss Functions**: LSGAN, Hinge Loss, and Vanilla GAN losses
- **Comprehensive Evaluation**: FID score calculation for quality assessment
- **Artifact Mitigation**: Solutions for checkerboard artifacts and black holes
- **Experiment Tracking**: Weights & Biases integration for monitoring training

## Architecture Details

### Generators

#### ResNet-based Generator
- **Encoder**: 3 downsampling blocks (64→128→256 channels)
- **Bottleneck**: 6 residual blocks with reflection padding
- **Decoder**: 2 upsampling blocks with configurable upsampling method
- **Features**: 
  - Reflection padding to prevent edge artifacts
  - Instance normalization for style transfer
  - GELU activation functions
  - Configurable dropout rates

#### UNet-based Generator
- **Architecture**: Skip connections between encoder and decoder
- **Depth**: 7 downsampling/upsampling layers
- **Features**:
  - Skip connections preserve spatial information
  - Gradual channel progression (64→512→64)
  - Dropout in early decoder layers

### Discriminator
- **Type**: PatchGAN discriminator
- **Architecture**: 4 convolutional layers
- **Features**:
  - LeakyReLU activation
  - Instance normalization
  - No normalization in first layer for stability

## 🔧 Implementation Highlights

### Artifact Solutions
The implementation includes solutions for common GAN artifacts:

1. **Checkerboard Artifacts**: 
   - Option to use `ConvTranspose2d` with `kernel_size=4, stride=2, padding=1`
   - Alternative: Upsample + Conv with bilinear interpolation

2. **Black/White Holes**:
   - Reduced dropout in upsampling layers
   - Fixed InstanceNorm initialization

3. **Edge Fading**:
   - Reflection padding throughout the network
   - Proper normalization handling

### Training Features
- **Learning Rate Scheduling**: Linear decay after half the training epochs
- **Fake Sample Buffer**: Discriminator history buffer for stable training
- **Multiple Loss Components**:
  - Adversarial loss (LSGAN/Hinge/Vanilla)
  - Cycle consistency loss (L1)
  - Identity preservation loss (L1)

## Training Configuration

### Loss Weights
- **Cycle Consistency**: 10.0 (most important for reconstruction)
- **Identity Preservation**: 5.0 (maintains color distribution)
- **Adversarial**: 1.0 (generates realistic outputs)

## Usage

## Experiment Tracking

The project uses **Weights & Biases** for comprehensive experiment tracking:

- **Loss Curves**: Generator and discriminator losses
- **Validation Images**: Real-time generation examples
- **Model Architecture**: Automatic model summaries
- **Hyperparameter Logging**: Complete configuration tracking

### Key Metrics Tracked
- Generator Loss (Total, Identity, Cycle, Adversarial)
- Discriminator Loss
- Training Time per Epoch
- Validation Image Examples
- FID Scores

## Architectural Experiments

### `model.ipynb` - General Implementation

### `model_unet.ipynb` - UNet Experiments (Main working notebook)

### `model_old_arch.ipynb` - Legacy Experiments
