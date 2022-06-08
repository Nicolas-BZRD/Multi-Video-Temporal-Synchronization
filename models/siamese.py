import os
from tensorflow.keras import models, layers, backend, callbacks, metrics

class SiameseCNN():
    def __init__(self, dropout: list, img_shape: tuple, v=1) -> None:
        self.version = v
        if(v==1):
            self._v1(dropout, img_shape)
        elif(v==2):
            self._v2(dropout, img_shape)

    def _v1(self, dropout: list, img_shape: tuple):
        left_input = layers.Input(img_shape)
        right_input = layers.Input(img_shape)

        # Convolutional Neural Network
        cnn = models.Sequential()
        cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
        cnn.add(layers.MaxPooling2D((3, 3)))

        cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))
        cnn.add(layers.Dropout(dropout[0]))

        cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))
        cnn.add(layers.Dropout(dropout[1]))

        cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))
        cnn.add(layers.Dropout(dropout[2]))

        cnn.add(layers.GlobalAveragePooling2D())
        cnn.add(layers.Dense(48, activation='relu'))
        cnn.add(layers.Dropout(dropout[3]))

        # Generate the encodings (feature vectors) for the two images
        left_feats = cnn(left_input)
        right_fits = cnn(right_input)

        # Add a customized layer to compute the difference between the encodings
        difference_layer = layers.Lambda(lambda tensors:backend.abs(tensors[0] - tensors[1]))
        distance = difference_layer([left_feats, right_fits])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = layers.Dense(1,activation='sigmoid')(distance)

        # Connect the inputs with the outputs
        self.model = models.Model(inputs=[left_input,right_input],outputs=prediction)

        # Compile the model
        self.model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy", metrics.Recall(), metrics.Precision()])

    def _v2(self, dropout: list, img_shape: tuple):
        left_input = layers.Input(img_shape)
        right_input = layers.Input(img_shape)

        # Convolutional Neural Network
        cnn = models.Sequential()
        cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
        cnn.add(layers.MaxPooling2D((2, 2)))

        cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))
        cnn.add(layers.Dropout(dropout[0]))

        cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))

        cnn.add(layers.Conv2D(96, (1, 1), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))
        cnn.add(layers.Dropout(dropout[2]))

        cnn.add(layers.Conv2D(96, (3, 3), activation='relu'))
        cnn.add(layers.MaxPooling2D((2, 2)))

        cnn.add(layers.GlobalAveragePooling2D())
        cnn.add(layers.Dense(64, activation='relu'))
        cnn.add(layers.Dropout(dropout[3]))

        # Generate the encodings (feature vectors) for the two images
        left_feats = cnn(left_input)
        right_fits = cnn(right_input)

        # Add a customized layer to compute the difference between the encodings
        difference_layer = layers.Lambda(lambda tensors:backend.abs(tensors[0] - tensors[1]))
        distance = difference_layer([left_feats, right_fits])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = layers.Dense(1,activation='sigmoid')(distance)

        # Connect the inputs with the outputs
        self.model = models.Model(inputs=[left_input,right_input],outputs=prediction)

        # Compile the model
        self.model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy", metrics.Recall(), metrics.Precision()])
    
    def checkpointCallback(self, path):
        return callbacks.ModelCheckpoint(
            filepath=fr"{os.getcwd()}/{path}",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

    def loadWeights(self, path):
        self.model.load_weights(fr"{os.getcwd()}/{path}")

    def save(self, path):
        self.model.save(fr"{os.getcwd()}/{path}")