from glob import glob
import numpy as np
import os
import json
from keras.models import load_model
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, MaxPool2D, BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Custom encoder block used in U-Net model
class EncoderBlock(Layer):
    def __init__(self, filters, rate, pooling=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.c1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.drop = Dropout(rate)
        self.c2 = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu',
                         kernel_initializer='he_normal')
        self.pool = MaxPool2D()

    def call(self, X, **kwargs):
        x = self.c1(X)  # First convolution layer
        x = self.drop(x)  # Dropout layer for regularization
        x = self.c2(x)  # Second convolution layer
        if self.pooling:
            y = self.pool(x)  # Max pooling layer for down-sampling
            return y, x  # Return down-sampled feature map and original feature map for skip connection
        else:
            return x


# Custom decoder block used in U-Net model
class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X, **kwargs):
        X, skip_X = X  # Get input and corresponding skip connection
        x = self.up(X)  # Upsampling layer to increase spatial dimensions
        c_ = concatenate([x, skip_X])  # Concatenate upsampled input with skip connection
        x = self.net(c_)  # Pass concatenated feature map through another encoder block
        return x


# Custom attention gate used to improve focus on relevant features
class AttentionGate(Layer):
    def __init__(self, filters, bn, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.filters = filters
        self.bn = bn
        self.normal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.down = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                           kernel_initializer='he_normal')
        self.learn = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')
        self.resample = UpSampling2D()
        self.BN = BatchNormalization()
    def call(self, X, **kwargs):
        X, skip_X = X  # Get input and corresponding skip connection
        x = self.normal(X)  # Normal convolution layer
        skip = self.down(skip_X)  # Downsample skip connection
        x = Add()([x, skip])  # Add input and skip connection feature maps
        x = self.learn(x)  # Learn attention weights
        x = self.resample(x)  # Upsample attention map to original size
        f = Multiply()([x, skip_X])  # Multiply attention map with skip connection
        if self.bn:
            return self.BN(f)  # Apply batch normalization if specified
        else:
            return f


# Dictionary to ensure custom layers are loaded correctly
custom_objects = {'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock, 'AttentionGate': AttentionGate}


# Function to load configuration settings from a JSON file
def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


# Load configuration settings
config = load_config("config.json")


# Function to load and preprocess test data
def load_test_data(image_dir, mask_dirs, image_size):
    images, masks = [], []
    for mask_dir in mask_dirs:
        image_files = glob(os.path.join(image_dir, mask_dir, '*.png'))  # Get all image files in directory
        for image_file in image_files:
            if '_mask' in image_file:
                continue  # Skip mask files
            image = load_img(image_file, target_size=(image_size, image_size), color_mode='rgb')  # Load image
            image = img_to_array(image)  # Convert image to array
            images.append(image / 255.0)  # Normalize image
            mask_file = image_file.replace('.png', '_mask.png')  # Get corresponding mask file
            mask = load_img(mask_file, target_size=(image_size, image_size), color_mode='grayscale')  # Load mask
            mask = img_to_array(mask)[:, :, 0]  # Convert mask to array and remove color channel dimension
            mask = np.expand_dims(mask, axis=-1)  # Expand dimensions to add a new axis
            mask = (mask > 127).astype(np.float32)  # Binarize mask
            masks.append(mask)
    return np.array(images), np.array(masks)  # Return arrays of images and masks


# Load pre-trained U-Net model with custom layers
unet_model = load_model(config["unet_model_path"], custom_objects=custom_objects)

# Define constants for data loading
IMAGE_DIR = 'C:\\Users\\AhmetSahinCAKIR\\Desktop\\Ahmet\\Bitirme\\Dataset_BUSI_with_GT'
MASK_DIRS = ['benign', 'malignant', 'normal']
IMAGE_SIZE = 256

# Load and preprocess all test images and masks
all_images, all_masks = load_test_data(IMAGE_DIR, MASK_DIRS, IMAGE_SIZE)
all_images, all_masks = all_images.astype('float32'), all_masks.astype('float32')

# Split data into testing sets
_, test_images, _, test_masks = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

# Make predictions on the test images using the U-Net model
predicted_masks = unet_model.predict(test_images, batch_size=5)


# Function to calculate the accuracy of predicted masks
def calculate_accuracy(true_masks, pred_masks):
    true_masks = true_masks.flatten()  # Flatten true masks to 1D
    pred_masks = (pred_masks > 0.5).flatten()  # Apply threshold to predicted masks and flatten to 1D
    accuracy = accuracy_score(true_masks, pred_masks)  # Calculate accuracy
    return accuracy * 100


# Calculate and print the accuracy of the U-Net model
accuracy = calculate_accuracy(test_masks, predicted_masks)
print(f'Accuracy of the model: {accuracy:.4f}')
