# 🧠 Image Inpainting – Restoring Damaged Areas Using Deep Learning (Pix2Pix)

This project focuses on **restoring damaged or missing regions in images** using a **deep learning–based inpainting model** built on the **Pix2Pix architecture**. It was developed as part of the *Image Processing* course

---

## 🎯 Project Overview

Image inpainting aims to reconstruct lost or deteriorated parts of images in a visually plausible way.  
Traditional approaches rely on texture propagation and diffusion methods (like Telea or Navier–Stokes), but in this project, we employ a **GAN-based approach (Pix2Pix)** to learn the mapping between **damaged images and their restored versions**.

---

## ⚙️ Methodology

1. **Dataset Preparation**  
   - Input images were synthetically damaged by applying random masks.  
   - Paired data `(damaged, original)` were used to train the model.

2. **Model Architecture**  
   - Generator: U-Net–style encoder–decoder network.  
   - Discriminator: PatchGAN that evaluates local realism.  
   - Loss: Combination of adversarial loss and L1 reconstruction loss.

3. **Training & Evaluation**  
   - Framework: TensorFlow / Keras  
   - Epochs and performance tracked with `history.pkl`.  
   - Evaluated using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)** metrics.

---

## 📁 Project Structure

| File Name | Description |
|------------|-------------|
| `model.py` | Defines the Pix2Pix model |
| `pix2pix_model.h5` | Trained model weights |
| `PSNR & SSIM.py` | Evaluation metrics |
| `test.py` | Testing / inference script |
| `history.pkl` | Saved training history |

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/image-inpainting.git
   cd image-inpainting
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run inference**
   ```bash
   python test.py

6. **Evaluate results**
   ```bash
   python "PSNR & SSIM.py"

---

## 📊 Results

  - The trained model successfully reconstructs missing image regions, producing visually realistic restorations.
  - Quantitative results show improvements in PSNR and SSIM over traditional inpainting techniques.

---

## 🧠 Key Learnings

  - Understanding GAN-based architectures for image translation tasks.
  - Comparing traditional OpenCV methods with deep learning models.
  - Implementing PSNR and SSIM to quantify image restoration quality.

---

## 🌟 Contributors

<a href="https://github.com/JanithM">
  <img src="https://avatars.githubusercontent.com/JanithM" width="80" style="border-radius: 50%;" />
</a>
<a href="https://github.com/kavindu016">
  <img src="https://avatars.githubusercontent.com/kavindu016" width="80" style="border-radius: 50%;" />
</a>
<a href="https://github.com/NethminE20">
  <img src="https://avatars.githubusercontent.com/NethminE20" width="80" style="border-radius: 50%;" />
</a>
<a href="https://github.com/github-username4">
  <img src="https://avatars.githubusercontent.com/github-username4" width="80" style="border-radius: 50%;" />
</a>
---


