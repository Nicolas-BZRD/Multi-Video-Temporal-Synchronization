import os, sys
import random

from tensorflow.keras import models, layers, backend, metrics
import cv2 as cv
import numpy as np

# --------------------------------------------------------------------------

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# --------------------------------------------------------------------------

img_shape = (224,224,3)
dropout = [0,0,0,0]

left_input = layers.Input(img_shape)
right_input = layers.Input(img_shape)

# Convolutional Neural Network
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
cnn.add(layers.MaxPooling2D((3, 3)))
cnn.add(layers.Dropout(dropout[0]))

cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Dropout(dropout[1]))

cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Dropout(dropout[2]))

cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Dropout(dropout[3]))

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
model = models.Model(inputs=[left_input,right_input],outputs=prediction)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy", metrics.Recall(), metrics.Precision()])

model.load_weights(r"2022-06-08 (cv_flow_full)/17-31-04/weight/weights")

# --------------------------------------------------------------------------

path = r"ISIA_cv_flow/1/"
images = os.listdir(fr"{path}/left")
numberImages = len(images)

ds = []

for i in range(7000):
  if i%50 == 0: print(i)
  if i%500 == 0: np.save(f"ds{i}", ds)

  y = random.randint(0,20)
      
  firstImageL = random.randint(0,numberImages-(y*2)-1)
  firstImageR = firstImageL+y

  imageL = []
  imageR = []
  try:
    with HiddenPrints():
      for j in range(20):
        tmp = cv.resize(cv.imread(fr"{path}/left/{firstImageL+j}.jpg"),(224,224))
        tmp = tmp/255
        imageL.append(tmp.reshape(1,224,224,3))

        tmp = cv.resize(cv.imread(fr"{path}/right/{firstImageR+j}.jpg"),(224,224))
        tmp = tmp/255
        imageR.append(tmp.reshape(1,224,224,3))

      result = []
      for w in range(0,20):
        for h in range(0,20):
          result.append(model.predict([(imageL[w], imageR[h])])[0][0])

      y = -1 if y == 20 else y
      ds.append([result, y, firstImageL, firstImageR])
  except:
    pass