import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths to your dataset
original_data_path = "landscape Images/color"  # Folder with clean images
damaged_data_path = "landscape Images/gray"  # Folder with corresponding damaged images

# Load and preprocess images
original_images = []
damaged_images = []

image_size = (128, 128)  # Resize if necessary

# Load images and normalize
for file_name in os.listdir(original_data_path):
    try:
        original_img = load_img(os.path.join(original_data_path, file_name), target_size=image_size)
        damaged_img = load_img(os.path.join(damaged_data_path, file_name), target_size=image_size)

        original_images.append(img_to_array(original_img) / 255.0)  # Normalize
        damaged_images.append(img_to_array(damaged_img) / 255.0)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Convert lists to numpy arrays
original_images = np.array(original_images)
damaged_images = np.array(damaged_images)

# Split into training and validation sets
train_original, val_original, train_damaged, val_damaged = train_test_split(
    original_images, damaged_images, test_size=0.2, random_state=42
)

# Load the trained model
pix2pix_model = load_model('pix2pix_model.h5')

# Function to load and preprocess a custom image
def preprocess_custom_image(image_path, target_size=(128, 128)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert it to a numpy array
    img_array = img_to_array(img) / 255.0  # Normalize the image
    # Expand dimensions to match the model input shape (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to display the custom image and the generated output
def display_custom_image_and_generated_output(model, image_path):
    # Preprocess the custom image
    custom_image = preprocess_custom_image(image_path)
    
    # Generate the image using the model
    generated_image = model.predict(custom_image)[0]  # Remove the batch dimension (1, 128, 128, 3) -> (128, 128, 3)
    
    # Display the custom image and the generated image side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the custom (damaged) image
    axes[0].imshow(custom_image[0])  # custom_image[0] removes the batch dimension (1, 128, 128, 3) -> (128, 128, 3)
    axes[0].set_title("Custom Image")
    axes[0].axis('off')
    
    # Ensure the generated image has the correct shape before displaying
    generated_image = np.squeeze(generated_image)  # Remove any remaining dimensions (1, 128, 128, 3) -> (128, 128, 3)
    
    # Display the generated image
    axes[1].imshow(generated_image)
    axes[1].set_title("Generated Image")
    axes[1].axis('off')
    
    plt.show()

# Example usage:
custom_image_path = "76.jpg"  # Replace with your custom image path
display_custom_image_and_generated_output(pix2pix_model, custom_image_path)
