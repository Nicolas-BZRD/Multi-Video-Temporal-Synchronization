import os
from tensorflow.keras import models, layers, backend, callbacks, metrics

class FeauturesExtraction():
    def __init__(self) -> None:
        img_shape = (224,224,3)
        dropout = [0,0,0,0]

        # Convolutional Neural Network
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Dropout(dropout[0]))

        self.model.add(layers.Conv2D(48, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(dropout[1]))

        self.model.add(layers.Conv2D(48, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(dropout[2]))

        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(dropout[3]))

        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(48, activation='relu'))
        
        self.model.load_weights(fr"{os.getcwd()}/models\cnn\weights\weights")

    def extract(self, img):
        return self.model.predict([img.reshape(1,224,224,3)])