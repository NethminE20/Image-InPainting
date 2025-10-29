import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model("pix2pix_model.h5")

# Paths to test images
test_damaged_path = "landscape Images/gray/13.jpg"
test_original_path = "landscape Images/color/13.jpg"

# Load and preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize to match model input
    img = img_to_array(img) / 255.0  # Normalize
    return img

damaged_img = preprocess_image(test_damaged_path)
original_img = preprocess_image(test_original_path)

# Expand batch dimension for model prediction
damaged_img_input = np.expand_dims(damaged_img, axis=0)

# Generate restored image
restored_img = np.squeeze(model.predict(damaged_img_input)[0])

# Ensure all images have the same shape
print(f"Original Image Shape: {original_img.shape}")
print(f"Restored Image Shape: {restored_img.shape}")

# Resize restored image to match original size (if needed)
if restored_img.shape != original_img.shape:
    from skimage.transform import resize
    restored_img = resize(restored_img, original_img.shape, anti_aliasing=True)

# Compute PSNR & SSIM
psnr_value = psnr(original_img, restored_img)
ssim_value = ssim(original_img, restored_img, data_range=1, win_size=11, channel_axis=-1)


print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.2f}")

# Display images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(damaged_img) 
axes[0].set_title("Damaged Image")
axes[1].imshow(restored_img)
axes[1].set_title("Restored Image")
axes[2].imshow(original_img)
axes[2].set_title("Original Image")
plt.show()