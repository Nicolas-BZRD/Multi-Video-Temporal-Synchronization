# -*- coding: utf-8 -*-
"""Encoder Decoder CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p1n74shb4Hoiv9I3aTMia_4ItkUe6uDW

# Encoder Decoder CNN notebook
"""

import tensorflow as tf
from matplotlib import pyplot as plt

"""## Data"""

# !unzip "/content/drive/MyDrive/ISIA Lab/Dataset/ISIA_cv_flow.zip"
# !unzip "/content/drive/MyDrive/ISIA Lab/Dataset/CLICK_cv_flow.zip"

"""### Train"""

left = tf.keras.utils.image_dataset_from_directory(
  r"ISIA_cv_flow/1/left",
  color_mode="rgb",
  image_size=(224,224),
  batch_size=None,
  labels=None,
  shuffle=False
)
left = left.map(lambda x: x/255)

right = tf.keras.utils.image_dataset_from_directory(
  r"ISIA_cv_flow/1/right",
  color_mode="rgb",
  image_size=(224,224),
  batch_size=None,
  labels=None,
  shuffle=False
)
right = right.map(lambda x: x/255)

left = tf.data.Dataset.zip((left, left))
right = tf.data.Dataset.zip((right, right))
ds_train = left.concatenate(right)

ds_train = ds_train.shuffle(1000).batch(32).prefetch(1)

"""### Validation"""

left = tf.keras.utils.image_dataset_from_directory(
  r"CLICK_cv_flow/1/left",
  color_mode="rgb",
  image_size=(224,224),
  batch_size=None,
  labels=None,
  shuffle=False
)
left = left.map(lambda x: x/255)

right = tf.keras.utils.image_dataset_from_directory(
  r"CLICK_cv_flow/1/right",
  color_mode="rgb",
  image_size=(224,224),
  batch_size=None,
  labels=None,
  shuffle=False
)
right = right.map(lambda x: x/255)

left = tf.data.Dataset.zip((left, left))
right = tf.data.Dataset.zip((right, right))
ds_val = left.concatenate(right)

ds_val = ds_val.take(2000).batch(32)

"""## Model"""

model = tf.keras.models.Sequential()

############
# Encoding #
############

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(224,224,3), padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
# model.add(tf.keras.layers.SpatialDropout2D(0.35))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
# model.add(tf.keras.layers.SpatialDropout2D(0.4))

model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
# model.add(tf.keras.layers.SpatialDropout2D(0.45))


# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1024, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))

# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(128, activation='relu'))


# ############
# # Decoding #
# ############

# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(1024, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(3136, activation='relu'))
# model.add(tf.keras.layers.Reshape((14,14,16)))

model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.UpSampling2D((2, 2)))
# model.add(tf.keras.layers.SpatialDropout2D(0.34))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.UpSampling2D((2, 2)))
# model.add(tf.keras.layers.SpatialDropout2D(0.35))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.UpSampling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.UpSampling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

"""## Fit"""

checkPoint =  tf.keras.callbacks.ModelCheckpoint(
  filepath=fr"weights",
  save_weights_only=True,
  monitor='val_accuracy',
  mode='max',
  save_best_only=True)

model.fit(ds_train, epochs=100, validation_data=ds_val, callbacks=checkPoint)

# """## Test"""

# take = 3

# ds_val = ds_val.unbatch().batch(1)
# images_pred = model.predict(ds_val.take(take))

# for images, labels in ds_val.take(take):
#   plt.imshow(images[0][:,:,:].numpy(), interpolation='nearest')
#   plt.show()

# plt.imshow(images_pred[take-2], interpolation='nearest')
# plt.show()