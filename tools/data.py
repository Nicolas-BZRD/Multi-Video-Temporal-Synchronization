import os
from numpy import number
import tensorflow as tf
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, db) -> None:
        path = fr"{os.getcwd()}/{db.path}"
        numberImages = len(os.listdir(fr"{path}/0/left"))

        # Get data from all the directories
        mode = "grayscale" if db.grayscale else 'rgb'
        ds = [] #0->leftWrong, 1->rightWrong, 2->leftPair, 3->rightPair
        for label in range(0,2):
            for side in ['left', 'right']:
                tmp = tf.keras.utils.image_dataset_from_directory(
                    fr"{path}/{label}/{side}",
                    color_mode=mode,
                    image_size=(db.img_heigth, db.img_width),
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

        # Create the dataset
        self.size = numberImages*2
        self.shape = (db.img_heigth, db.img_width,1) if db.grayscale else (db.img_heigth, db.img_width,3)
        self.data = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])


    def splitData(self, train_size=0.8, validation_size=0.1, test_size=0.1):
        ds_train=self.data.take(int(self.size*train_size))
        tmp=self.data.skip(int(self.size*train_size))

        tmp_size = self.size - int(self.size*train_size)

        ds_validation = tmp.take(int(tmp_size*validation_size))
        ds_test = tmp.skip(int(tmp_size*validation_size))

        ds_train = ds_train.shuffle(300).batch(64)
        ds_validation = ds_validation.batch(64)
        ds_test = ds_test.batch(64)

        return ds_train, ds_validation, ds_test


    def showImages(self, number = 10):
        for images, labels in self.data.take(number):
            f, axarr = plt.subplots(1,2)
            f.suptitle(labels.numpy())
            axarr[0].imshow(images[0][:,:,0].numpy(), cmap='gray', vmin=0, vmax=1)
            axarr[1].imshow(images[1][:,:,0].numpy(), cmap='gray', vmin=0, vmax=1)