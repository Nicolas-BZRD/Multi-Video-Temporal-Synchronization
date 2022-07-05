import os
from tensorflow.keras import applications, models, layers, backend, metrics, callbacks

class SiameseCNN():
    def __init__(self, img_shape: tuple) -> None:
        left_input = layers.Input(img_shape)
        right_input = layers.Input(img_shape)

        # Load efficient net
        efficient = applications.EfficientNetB2(
            include_top=False,
            input_shape=img_shape,
        )
        efficient.trainable=False

        # Convolutional Neural Network
        cnn = models.Sequential()
        cnn.add(efficient)
        cnn.add(layers.GlobalAveragePooling2D())
        cnn.add(layers.Dense(48, activation='relu'))

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

    def tensorboardCallback(self, LOG_PATH):
        return callbacks.TensorBoard(log_dir=LOG_PATH, histogram_freq=1)

    def loadWeights(self, path):
        self.model.load_weights(fr"{os.getcwd()}/{path}")

    def save(self, path):
        self.model.save(fr"{os.getcwd()}/{path}")