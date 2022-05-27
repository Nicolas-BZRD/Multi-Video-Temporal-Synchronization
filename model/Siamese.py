from tensorflow.keras import models, layers, backend

def SiameseCNN(img_shape=(224, 224, 1)):
    # Define the tensors for the two input images
    left_input = layers.Input(img_shape)
    right_input = layers.Input(img_shape)

    # Convolutional Neural Network
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
    cnn.add(layers.MaxPooling2D((3, 3)))

    cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Dropout(0.40))

    cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Dropout(0.50))

    cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Dropout(0.65))

    cnn.add(layers.GlobalAveragePooling2D())
    cnn.add(layers.Dense(48, activation='relu'))
    cnn.add(layers.Dropout(0.8))

    # Generate the encodings (feature vectors) for the two images
    left_feats = cnn(left_input)
    right_fits = cnn(right_input)

    # Add a customized layer to compute the difference between the encodings
    difference_layer = layers.Lambda(lambda tensors:backend.abs(tensors[0] - tensors[1]))
    distance = difference_layer([left_feats, right_fits])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = layers.Dense(1,activation='sigmoid')(distance)

    # Connect the inputs with the outputs
    model = models.Model(inputs=[left_input,right_input],outputs=prediction)

    # Compil
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

    return model