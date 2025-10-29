import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle  # For saving history

# Paths to your dataset
original_data_path = "landscape Images/color"  # Folder with clean images
damaged_data_path = "landscape Images/gray"  # Folder with corresponding damaged images

# Load and preprocess images
original_images = []
damaged_images = []

image_size = (128, 128)  # Resize if necessary

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

print(f"Loaded {len(original_images)} images. Training on {len(train_original)}, validating on {len(val_original)}.")

# Build the generator model
def build_generator():
    inputs = layers.Input(shape=(128, 128, 3))  # Start with 128x128 input images

    # Encoder (downsampling)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer

    # Decoder (upsampling)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample to 128x128
    x = layers.Dropout(0.3)(x)  # Dropout layer
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample to 128x128
    x = layers.Dropout(0.3)(x)  # Dropout layer

    # Output
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

# Build the discriminator model
def build_discriminator():
    inputs = layers.Input(shape=(128, 128, 3))  # Expect 128x128 RGB image

    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer

    # Flatten and output layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)  # Dropout layer
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, x)
    return model

# Build the Pix2Pix model (generator + discriminator)
def build_pix2pix_model(generator, discriminator):
    # Create the generator output
    gen_output = generator.output  # Output from the generator

    # Pass the generator output through the discriminator
    disc_output = discriminator(gen_output)

    # Create the model
    model = models.Model(inputs=generator.input, outputs=[gen_output, disc_output])
    return model

# Build the generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Build the Pix2Pix model
pix2pix_model = build_pix2pix_model(generator, discriminator)

# Compile the model with metrics for both outputs
pix2pix_model.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss=['mean_squared_error', 'binary_crossentropy'],
    metrics=['accuracy', 'accuracy']  # Metrics for both generator and discriminator
)

# Train the model
history = pix2pix_model.fit(
    train_damaged, [train_original, np.ones(len(train_original))],  # Generator input, target original images and label 1 for real images
    epochs=20,
    batch_size=16,
    validation_data=(val_damaged, [val_original, np.ones(len(val_original))])
)

# Save the trained model
pix2pix_model.save("pix2pix_model.h5")
print("Model training complete and saved.")

# Save the training history
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("History object saved successfully.")
