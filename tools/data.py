import os
import tensorflow as tf
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, path) -> None:
        path = fr"{path}"
        numberImages = len(os.listdir(fr"{path}/0/left"))

        # Create a wrong pair of images dataset.
        leftWrong = tf.keras.utils.image_dataset_from_directory(fr"{path}/0/left", labels=None, color_mode="grayscale",
        batch_size=None, image_size=(224, 224), shuffle=False)

        rightWrong = tf.keras.utils.image_dataset_from_directory(fr"{path}/0/right", labels=None, color_mode="grayscale",
        batch_size=None, image_size=(224, 224), shuffle=False)

        wrong = tf.data.Dataset.zip((leftWrong, rightWrong))

        # Create a dataset of correct pair images.
        leftPair = tf.keras.utils.image_dataset_from_directory(fr"{path}/1/left", labels=None, color_mode="grayscale",
        batch_size=None,image_size=(224, 224),shuffle=False)

        rightPair = tf.keras.utils.image_dataset_from_directory(fr"{path}/1/right", labels=None, color_mode="grayscale",
        batch_size=None, image_size=(224, 224), shuffle=False)

        pair = tf.data.Dataset.zip((leftPair, rightPair))

        # Create labels
        wrongLables = tf.zeros([numberImages])
        wrongLables = tf.data.Dataset.from_tensor_slices(wrongLables)
        wrong = tf.data.Dataset.zip((wrong, wrongLables))

        pairLabels = tf.ones([numberImages])
        pairLabels = tf.data.Dataset.from_tensor_slices(pairLabels)
        pair = tf.data.Dataset.zip((pair, pairLabels))

        # Concatenate dataset with a radom shuffle
        self.dataset_size = numberImages*2
        self.dataset = tf.data.Dataset.sample_from_datasets([pair, wrong], weights=[0.5, 0.5])


    def splitData(self, train_size=0.8, test_size=0.2):
        ds_train=self.dataset.take(int(self.dataset_size*train_size))
        ds_test=self.dataset.skip(int(self.dataset_size*test_size))

        ds_train = ds_train.shuffle(300).batch(64)

        return ds_train, ds_test


    def showImages(self, number = 10):
        for images, labels in self.dataset.take(number):
            f, axarr = plt.subplots(1,2)
            f.suptitle(labels.numpy())
            axarr[0].imshow(images[0][:,:,0].numpy().astype("uint8"), cmap='gray')
            axarr[1].imshow(images[1][:,:,0].numpy().astype("uint8"), cmap='gray')