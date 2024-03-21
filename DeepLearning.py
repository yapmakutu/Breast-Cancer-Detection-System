import cv2
import numpy as np
from keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate, Add, Multiply, MaxPool2D, BatchNormalization
from keras.models import load_model


# Özelleştirilmiş katmanlar
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


class DeepLearning:
    def __init__(self, unet_model_path, cnn_model_path):
        self.segmentation_model = load_model(unet_model_path,
                                             custom_objects={'EncoderBlock': EncoderBlock, 'DecoderBlock': DecoderBlock,
                                                             'AttentionGate': AttentionGate})
        self.classification_model = load_model(cnn_model_path)

    def segment_and_classify(self, image_path):
        test_image = cv2.imread(image_path)
        resized_image = cv2.resize(test_image, (256, 256))
        resized_image = np.expand_dims(resized_image, axis=0)

        predicted_mask = self.segmentation_model.predict(resized_image)
        predicted_mask = cv2.resize(predicted_mask[0], (test_image.shape[1], test_image.shape[0]))
        _, binary_mask = cv2.threshold(predicted_mask, 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = (binary_mask * 255).astype(np.uint8)

        segmented_pixels = np.sum(binary_mask > 0)
        total_pixels = binary_mask.size
        segmented_ratio = (segmented_pixels / total_pixels) * 100

        # Eğer segmentlenmiş piksel oranı %2'den az ise "Normal (No cancer)" sonucunu döndür
        if segmented_ratio < 1:
            diagnosis = self.update_diagnosis("Normal")
        else:
            predicted_mask_resized = cv2.resize(binary_mask, (256, 256))
            predicted_mask_final = np.expand_dims(predicted_mask_resized, axis=0)
            predicted_mask_final = np.expand_dims(predicted_mask_final, axis=-1)
            prediction = self.classification_model.predict(predicted_mask_final)
            diagnosis = self.update_diagnosis(prediction)

        return binary_mask, diagnosis

    def update_diagnosis(self, prediction):
        if prediction == "Normal":
            diagnosis_text = "Normal (No cancer)"
        elif isinstance(prediction, np.ndarray) and prediction.shape[-1] == 1:
            diagnosis_text = "Malignant" if prediction[0][0] > 0.6 else "Benign"
        else:
            diagnosis_text = "Not Cancer"
        return diagnosis_text