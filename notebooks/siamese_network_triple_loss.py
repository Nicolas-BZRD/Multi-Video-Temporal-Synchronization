# -*- coding: utf-8 -*-
"""siamese_network_triple_loss.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QJbJLHytXdhkuOVJahVfkb0sacVUvAk3

# Image similarity estimation using a Siamese Network with a triplet loss

## Setup
"""

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, metrics, callbacks, backend

"""## Load the dataset

We are going to load the *Totally Looks Like* dataset and unzip it inside the `~/.keras` directory
in the local environment.

The dataset consists of two separate files:

* `left.zip` contains the images that we will use as the anchor.
* `right.zip` contains the images that we will use as the positive sample (an image that looks like the anchor).
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !unzip "/content/drive/MyDrive/ISIA Lab/Dataset/ISIA.zip"
# !unzip "/content/drive/MyDrive/ISIA Lab/Dataset/CLICK.zip"
# !unzip "/content/drive/MyDrive/ISIA Lab/Dataset/robot.zip"

"""## Preparing the data

We are going to use a `tf.data` pipeline to load the data and generate the triplets that we
need to train the Siamese network.

We'll set up the pipeline using a zipped list with anchor, positive, and negative filenames as
the source. The pipeline will load and preprocess the corresponding images.

Let's setup our data pipeline using a zipped list with an anchor, positive,
and negative image filename as the source. The output of the pipeline
contains the same triplet with every image loaded and preprocessed.
"""

path = fr"ISIA/images"

# Get data from all the directories
ds = [] #0->leftWrong same as leftPair, 1->rightWrong, 2->leftPair, 3->rightPair
for label in ['wrong', 'good']:
    for side in ['left', 'right']:
        tmp = tf.keras.utils.image_dataset_from_directory(
            fr"{path}/{label}/{side}",
            color_mode="rgb",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        tmp = tmp.map(lambda x: x/255)
        ds.append(tmp)

negative_input = ds[1]
anchor_input = ds[2]
positive_input = ds[3]

train_dataset = tf.data.Dataset.zip((anchor_input, positive_input, negative_input))

size = len(os.listdir(fr"{path}/good/left"))
train_size=0.8

val_dataset = train_dataset.skip(int(size*train_size))
train_dataset = train_dataset.take(int(size*train_size))

train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(1)
val_dataset = val_dataset.batch(64).prefetch(1)

print(train_dataset)
print(val_dataset)

path = fr"CLICK/images"

# Get data from all the directories
ds = [] #0->leftWrong same as leftPair, 1->rightWrong, 2->leftPair, 3->rightPair
for label in ['wrong', 'good']:
    for side in ['left', 'right']:
        tmp = tf.keras.utils.image_dataset_from_directory(
            fr"{path}/{label}/{side}",
            color_mode="rgb",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        tmp = tmp.map(lambda x: x/255)
        ds.append(tmp)

negative_input = ds[1]
anchor_input = ds[2]
positive_input = ds[3]

test_dataset = tf.data.Dataset.zip((anchor_input, positive_input, negative_input))
test_dataset = test_dataset.batch(1).prefetch(1)
print(test_dataset)

"""## Setting up the embedding generator model"""

# Convolutional Neural Network
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
# cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D((3, 3)))

cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
# cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D((2, 2)))

cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
# cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D((2, 2)))

cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
# cnn.add(layers.BatchNormalization())
cnn.add(layers.MaxPooling2D((2, 2)))

cnn.add(layers.GlobalAveragePooling2D())
cnn.add(layers.Dense(48, activation='relu'))

cnn.summary()

"""## Setting up the Siamese Network model

The Siamese network will receive each of the triplet images as an input,
generate the embeddings, and output the distance between the anchor and the
positive embedding, as well as the distance between the anchor and the negative
embedding.

To compute the distance, we can use a custom layer `DistanceLayer` that
returns both values as a tuple.
"""

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=(224,224,3))
positive_input = layers.Input(name="positive", shape=(224,224,3))
negative_input = layers.Input(name="negative", shape=(224,224,3))

distances = DistanceLayer()(
    cnn(anchor_input),
    cnn(positive_input),
    cnn(negative_input),
)

siamese_network = models.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

"""## Putting everything together

We now need to implement a model with custom training loop so we can compute
the triplet loss using the three embeddings produced by the Siamese network.

Let's create a `Mean` metric instance to track the loss of the training process.
"""

class SiameseModel(models.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(???f(A) - f(P)????? - ???f(A) - f(N)????? + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))

"""## Training"""

checkpoint = callbacks.ModelCheckpoint(
  filepath=fr"weights/weights",
  save_weights_only=True,
  monitor='val_loss',
  mode='min',
  save_best_only=True)

siamese_model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint])

"""## Evaluate"""

siamese_model.load_weights("weights/weights")

siamese_model.evaluate(test_dataset)

# import shutil
# shutil.make_archive('/content/drive/MyDrive/weights', 'zip', '/content/weights')

"""## Deep identical images

### Data
"""

path = fr"ISIA/images"
numberImages = len(os.listdir(fr"{path}/good/left"))

# Get data from all the directories
ds = [] #0->leftWrong, 1->rightWrong, 2->leftPair, 3->rightPair
for label in ['wrong', 'good']:
    for side in ['left', 'right']:
        tmp = tf.keras.utils.image_dataset_from_directory(
            fr"{path}/{label}/{side}",
            color_mode="rgb",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        tmp = tmp.map(lambda x: x/255)
        ds.append(tmp)

wrong = tf.data.Dataset.zip((ds[0], ds[1]))
pair = tf.data.Dataset.zip((ds[2], ds[3]))

# Create labels
wrongLables = tf.zeros([numberImages])
wrongLables = tf.data.Dataset.from_tensor_slices(wrongLables)
wrong = tf.data.Dataset.zip((wrong, wrongLables))

pairLabels = tf.ones([numberImages])
pairLabels = tf.data.Dataset.from_tensor_slices(pairLabels)
pair = tf.data.Dataset.zip((pair, pairLabels))

train_dataset = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])

size = len(os.listdir(fr"{path}/good/left"))*2
train_size=0.8

val_dataset = train_dataset.skip(int(size*train_size))
train_dataset = train_dataset.take(int(size*train_size))

train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(1)
val_dataset = val_dataset.batch(64)

print(train_dataset)
print(val_dataset)

"""### Test Data
"""

path = fr"CLICK/images"
numberImages = len(os.listdir(fr"{path}/good/left"))

# Get data from all the directories
ds = [] #0->leftWrong, 1->rightWrong, 2->leftPair, 3->rightPair
for label in ['wrong', 'good']:
    for side in ['left', 'right']:
        tmp = tf.keras.utils.image_dataset_from_directory(
            fr"{path}/{label}/{side}",
            color_mode="rgb",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        tmp = tmp.map(lambda x: x/255)
        ds.append(tmp)

wrong = tf.data.Dataset.zip((ds[0], ds[1]))
pair = tf.data.Dataset.zip((ds[2], ds[3]))

# Create labels
wrongLables = tf.zeros([numberImages])
wrongLables = tf.data.Dataset.from_tensor_slices(wrongLables)
wrong = tf.data.Dataset.zip((wrong, wrongLables))

pairLabels = tf.ones([numberImages])
pairLabels = tf.data.Dataset.from_tensor_slices(pairLabels)
pair = tf.data.Dataset.zip((pair, pairLabels))

test_click = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])

test_click = test_click.batch(1)

### Robot dataset

path = fr"robot/images"
numberImages = len(os.listdir(fr"{path}/good/left"))

# Get data from all the directories
ds = [] #0->leftWrong, 1->rightWrong, 2->leftPair, 3->rightPair
for label in ['wrong', 'good']:
    for side in ['left', 'right']:
        tmp = tf.keras.utils.image_dataset_from_directory(
            fr"{path}/{label}/{side}",
            color_mode="rgb",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        tmp = tmp.map(lambda x: x/255)
        ds.append(tmp)

wrong = tf.data.Dataset.zip((ds[0], ds[1]))
pair = tf.data.Dataset.zip((ds[2], ds[3]))

# Create labels
wrongLables = tf.zeros([numberImages])
wrongLables = tf.data.Dataset.from_tensor_slices(wrongLables)
wrong = tf.data.Dataset.zip((wrong, wrongLables))

pairLabels = tf.ones([numberImages])
pairLabels = tf.data.Dataset.from_tensor_slices(pairLabels)
pair = tf.data.Dataset.zip((pair, pairLabels))

test_robot = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])

test_robot = test_robot.batch(1)

print(test_click)
print(test_robot)

"""#### Model"""

cnn.trainable = False

left_input = layers.Input((224,224,3))
right_input = layers.Input((224,224,3))

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

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy", metrics.Recall(), metrics.Precision()])

model.summary()

os.mkdir("weights_similarity")

checkpoint = callbacks.ModelCheckpoint(
  filepath=fr"weights_similarity/weights",
  save_weights_only=True,
  monitor='val_accuracy',
  mode='max',
  save_best_only=True)

model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint])



"""## Evaluate"""

model.load_weights("weights_similarity/weights")

loss, accuracy, recall, precision = model.evaluate(test_click)
f = ((1 + 1**2) * precision * recall) / ((1**2) * precision + recall)
print(f"loss: {loss}\naccuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\nf1 score: {f}")


loss, accuracy, recall, precision = model.evaluate(test_robot)
f = ((1 + 1**2) * precision * recall) / ((1**2) * precision + recall)
print(f"loss: {loss}\naccuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\nf1 score: {f}")


"""## Inspecting what the network has learned

At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.

We can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the
similarity between embeddings.

Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""

# sample = next(iter(train_dataset))
# visualize(*sample)

# anchor, positive, negative = sample
# anchor_embedding, positive_embedding, negative_embedding = (
#     embedding(resnet.preprocess_input(anchor)),
#     embedding(resnet.preprocess_input(positive)),
#     embedding(resnet.preprocess_input(negative)),
# )

"""Finally, we can compute the cosine similarity between the anchor and positive
images and compare it with the similarity between the anchor and the negative
images.

We should expect the similarity between the anchor and positive images to be
larger than the similarity between the anchor and the negative images.
"""

# cosine_similarity = metrics.CosineSimilarity()

# positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
# print("Positive similarity:", positive_similarity.numpy())

# negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
# print("Negative similarity", negative_similarity.numpy())