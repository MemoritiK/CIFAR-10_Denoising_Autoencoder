# STL-10 Image Denoising with Lightweight U-Net

A convolutional neural network trained to remove Gaussian noise from STL-10 images (96×96 RGB). The model uses a **14-layer U-Net architecture** with skip connections and residual learning to preserve spatial details while effectively denoising.

## Model Architecture

### **Encoder (5 conv layers)**
- **Level 1:** Conv(3 → 32) → Conv(32 → 32), MaxPool2d (96×96 → 48×48)
- **Level 2:** Conv(32 → 64) → Conv(64 → 64), MaxPool2d (48×48 → 24×24)
- **Level 3:** Conv(64 → 128), MaxPool2d (24×24 → 12×12)

### **Bottleneck (2 conv layers)**
- Conv(128 → 256) → Conv(256 → 256)

### **Decoder (6 conv layers)**
- **Level 3:** ConvTranspose2d(256 → 128) → Concat with e3 → Conv(256 → 128) → Conv(128 → 128)
- **Level 2:** ConvTranspose2d(128 → 64) → Concat with e2 → Conv(128 → 64) → Conv(64 → 64)
- **Level 1:** ConvTranspose2d(64 → 32) → Concat with e1 → Conv(64 → 32) → Conv(32 → 32)
- **Output:** 1×1 Conv(32 → 3) with residual connection

The architecture compresses images from **3×96×96 → 256×12×12** and reconstructs them back with skip connections preserving fine details.

## Key Features

- **Residual learning:** Predicts noise, subtracts from input: `output = x - noise_pred`
- **Skip connections:** Preserves spatial details across encoder-decoder levels
- **Gradient loss:** 0.1 weight to preserve edges and textures
- **Curriculum learning:** Noise level decreases from σ=0.3 → σ=0.1 during first 20 epochs

## Training Summary

- **Dataset:** STL-10 (5,000 train / 8,000 test images, 96×96 RGB)
- **Batch size:** 64
- **Epochs:** 60
- **Loss:** MSE + 0.1 × Gradient Loss
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR (T_max=60, eta_min=1e-6)
- **Gradient clipping:** 1.0

## Results
<img width="1777" height="3037" alt="image" src="https://github.com/user-attachments/assets/3fe2cb3e-9184-4bb6-a36d-16f3109df564" />
<img width="1482" height="730" alt="image" src="https://github.com/user-attachments/assets/04520748-ec82-4edf-a770-26eb0a548a41" />
<img width="1318" height="434" alt="image" src="https://github.com/user-attachments/assets/617d53ef-9cee-47ff-abe7-8d8a5915c39b" />

