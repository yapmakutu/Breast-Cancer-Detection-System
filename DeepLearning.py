import logging
import sys
import os
import tempfile
import cv2
import numpy as np
from keras.models import load_model
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, MaxPool2D, BatchNormalization

# Custom Layers
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


def load_unet_model(path):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, path)
        model = load_model(model_path, custom_objects={'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock,
                                                       'AttentionGate': AttentionGate})
        logging.info("U-Net model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading U-Net model: {e}")
        raise


def load_cnn_model(path):
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, path)
        model = load_model(model_path)
        logging.info("CNN model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading CNN model: {e}")
        raise


class DeepLearning:
    def __init__(self, unet_model_path, cnn_model_path):
        self.segmentation_model = load_unet_model(unet_model_path)
        self.classification_model = load_cnn_model(cnn_model_path)

    @staticmethod
    def preprocess_image(image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not loaded correctly")
            resized_image = cv2.resize(image, (256, 256))
            resized_image = np.expand_dims(resized_image, axis=0)
            return resized_image
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise

    @staticmethod
    def check_contiguity(mask):
        num_labels, labels_im = cv2.connectedComponents(mask)
        return num_labels

    def segment_and_classify(self, image_path):
        try:
            resized_image = self.preprocess_image(image_path)
            logging.debug("Image preprocessed successfully.")

            # Predict the segmentation mask
            predicted_mask = self.segmentation_model.predict(resized_image)
            predicted_mask_resized = cv2.resize(predicted_mask[0], (resized_image.shape[2], resized_image.shape[1]))
            logging.debug("Segmentation mask predicted successfully.")

            # Binary mask
            _, binary_mask = cv2.threshold(predicted_mask_resized, 0.5, 1, cv2.THRESH_BINARY)
            binary_mask = (binary_mask * 255).astype(np.uint8)
            logging.debug("Binary mask created successfully.")

            # Check contiguity
            num_labels = self.check_contiguity(binary_mask)
            logging.debug(f"Number of labels in mask: {num_labels}")

            if num_labels > 2:  # If more than one component exists, apply threshold
                segmented_pixels = np.sum(binary_mask > 0)
                total_pixels = binary_mask.size
                segmented_ratio = (segmented_pixels / total_pixels) * 100

                if segmented_ratio < 1.0:  # If segmented area is less than 1%
                    binary_mask.fill(0)  # Set the mask to black
                    segmented_image_path = os.path.join(tempfile.gettempdir(), "segmented_image.png")
                    if not cv2.imwrite(segmented_image_path, binary_mask):
                        raise ValueError(f"1 Could not write image to path: {segmented_image_path}")
                    return segmented_image_path, "No cancer"

            # Resize and prepare mask for classification
            predicted_mask_final = cv2.resize(binary_mask, (256, 256))
            predicted_mask_final = np.expand_dims(predicted_mask_final, axis=0)
            predicted_mask_final = np.expand_dims(predicted_mask_final, axis=-1)

            # Classification
            prediction = self.classification_model.predict(predicted_mask_final)
            diagnosis = "Malignant" if prediction[0][0] > 0.6 else "Benign"
            logging.debug(f"Prediction: {prediction}, Diagnosis: {diagnosis}")

            # Save mask
            segmented_image_path = os.path.join(tempfile.gettempdir(), "segmented_image.png")
            if not cv2.imwrite(segmented_image_path, binary_mask):
                raise ValueError(f"2 Could not write image to path: {segmented_image_path}")

            return segmented_image_path, diagnosis
        except Exception as e:
            logging.error(f"Error in segment_and_classify: {e}")
            raise
