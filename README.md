# CIFAR-10 Denoising Autoencoder (PyTorch)

A convolutional autoencoder trained to remove Gaussian noise from CIFAR-10 images.
The model uses a **3-layer encoder** and **3-layer decoder** with strided convolutions and transposed convolutions to learn a compact latent representation and reconstruct clean images.

## Model Architecture

### **Encoder**

* Conv2d(3 → 32), stride 2 → ReLU
* Conv2d(32 → 64), stride 2 → ReLU
* Conv2d(64 → 128), stride 2 → ReLU

### **Decoder**

* ConvTranspose2d(128 → 64), stride 2 → ReLU
* ConvTranspose2d(64 → 32), stride 2 → ReLU
* ConvTranspose2d(32 → 3), stride 2 → Sigmoid

The architecture compresses images from **3×32×32 → 128×4×4** and reconstructs them back.

## Training Summary

* **Batch size:** 64
* **Epochs:** 50
* **Loss:** MSE
* **Optimizer:** Adam (`lr = 1e-3`)
* **Noise model:** `image + N(0, 0.2²)`

## Results
<img width="864" height="886" alt="image" src="https://github.com/user-attachments/assets/6dfdf444-f580-43ef-943d-ba377694fc3d" />
<img width="584" height="455" alt="image" src="https://github.com/user-attachments/assets/62b8ebf9-ea4c-42f3-9baf-8e909442eb30" />
