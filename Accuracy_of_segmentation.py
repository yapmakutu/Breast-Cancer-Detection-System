from glob import glob
import numpy as np
import os
import json
from keras.models import load_model
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, MaxPool2D, BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Custom layers
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
        x = self.c1(X)
        x = self.drop(x)
        x = self.c2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x


class DecoderBlock(Layer):
    def __init__(self, filters, rate, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate
        self.up = UpSampling2D()
        self.net = EncoderBlock(filters, rate, pooling=False)

    def call(self, X, **kwargs):
        X, skip_X = X
        x = self.up(X)
        c_ = concatenate([x, skip_X])
        x = self.net(c_)
        return x


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
        X, skip_X = X
        x = self.normal(X)
        skip = self.down(skip_X)
        x = Add()([x, skip])
        x = self.learn(x)
        x = self.resample(x)
        f = Multiply()([x, skip_X])
        if self.bn:
            return self.BN(f)
        else:
            return f


custom_objects = {'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock, 'AttentionGate': AttentionGate}


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


config = load_config("config.json")


def load_test_data(image_dir, mask_dirs, image_size):
    images, masks = [], []
    for mask_dir in mask_dirs:
        image_files = glob(os.path.join(image_dir, mask_dir, '*.png'))
        for image_file in image_files:
            if '_mask' in image_file:
                continue  # Skip mask files
            image = load_img(image_file, target_size=(image_size, image_size), color_mode='rgb')
            image = img_to_array(image)
            images.append(image / 255.0)
            mask_file = image_file.replace('.png', '_mask.png')
            mask = load_img(mask_file, target_size=(image_size, image_size), color_mode='grayscale')
            mask = img_to_array(mask)[:, :, 0]
            mask = np.expand_dims(mask, axis=-1)
            mask = (mask > 127).astype(np.float32)
            masks.append(mask)
    return np.array(images), np.array(masks)


custom_objects = {'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock, 'AttentionGate': AttentionGate}
unet_model = load_model(config["unet_model_path"], custom_objects=custom_objects)

IMAGE_DIR = 'C:\\Users\\AhmetSahinCAKIR\\Desktop\\Ahmet\\Bitirme\\Dataset_BUSI_with_GT'
MASK_DIRS = ['benign', 'malignant', 'normal']
IMAGE_SIZE = 256

all_images, all_masks = load_test_data(IMAGE_DIR, MASK_DIRS, IMAGE_SIZE)
all_images, all_masks = all_images.astype('float32'), all_masks.astype('float32')

_, test_images, _, test_masks = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

# Becaouse of my pc memory, i decremented the batch size to 5
predicted_masks = unet_model.predict(test_images, batch_size=5)


def calculate_accuracy(true_masks, pred_masks):
    true_masks = true_masks.flatten()
    pred_masks = (pred_masks > 0.5).flatten()  # Applying threshold to convert probabilities to binary predictions
    accuracy = accuracy_score(true_masks, pred_masks)
    return accuracy * 100


accuracy = calculate_accuracy(test_masks, predicted_masks)
print(f'Accuracy of the model: {accuracy:.4f}')
