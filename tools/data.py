import os
from numpy import number
import tensorflow as tf
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path) -> None:
        path = fr"{path}"
        numberImages = len(os.listdir(fr"{path}/0/left"))

        # Get data from all the directories
        ds = [] #0->leftWrong, 1->rightWrong, 2->leftPair, 3->rightPair
        for label in range(0,2):
            for side in ['left', 'right']:
                tmp = tf.keras.utils.image_dataset_from_directory(
                    fr"{path}/{label}/{side}",
                    color_mode="grayscale",
                    image_size=(224, 224),
                    batch_size=None,
                    labels=None,
                    shuffle=False
                )
                tmp = tmp.map(lambda x: x/255)
                ds.append(tmp)

        wrong = tf.data.Dataset.zip((ds[0], ds[1]))
        pair = tf.data.Dataset.zip((ds[2], ds[3]))

        # Create labels
        wrongLables = tf.zeros([len(numberImages)])
        wrongLables = tf.data.Dataset.from_tensor_slices(wrongLables)
        wrong = tf.data.Dataset.zip((wrong, wrongLables))

        pairLabels = tf.ones([len(numberImages)])
        pairLabels = tf.data.Dataset.from_tensor_slices(pairLabels)
        pair = tf.data.Dataset.zip((pair, pairLabels))

        # Create the dataset
        self.dataset_size = numberImages*2
        self.dataset = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])


    def splitData(self, train_size=0.7, validation_size=0.15, test_size=0.15):
        ds_train=self.dataset.take(int(self.dataset_size*train_size))

        tmp_size = self.dataset_size - int(self.dataset_size*train_size)
        tmp=self.dataset.skip(int(self.dataset_size*train_size))

        ds_validation = tmp.take(int(tmp_size*validation_size))
        ds_test = tmp.skip(int(tmp_size*validation_size))

        ds_train = ds_train.shuffle(300).batch(64)

        return ds_train, ds_validation, ds_test


    def showImages(self, number = 10):
        for images, labels in self.dataset.take(number):
            f, axarr = plt.subplots(1,2)
            f.suptitle(labels.numpy())
            axarr[0].imshow(images[0][:,:,0].numpy().astype("uint8"), cmap='gray')
            axarr[1].imshow(images[1][:,:,0].numpy().astype("uint8"), cmap='gray')