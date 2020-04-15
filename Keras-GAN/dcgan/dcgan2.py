from sklearn.model_selection import train_test_split
import glob
import os

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import *
from keras.utils import np_utils

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.root_dir = '/Users/rasuharu/Dev/gan/image-collector/images/'
        self.class_names = os.listdir(self.root_dir)

    def load_imgs(self):

        img_paths = []
        labels = []
        images = []
        for cl_name in self.class_names:
            img_names = os.listdir(os.path.join(self.root_dir, cl_name))
            for img_name in img_names:
                img_paths.append(os.path.abspath(os.path.join(self.root_dir, cl_name, img_name)))
                hot_cl_name = self.get_class_one_hot(cl_name)
                labels.append(hot_cl_name)

        for img_path in img_paths:
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                import ipdb
                ipdb.set_trace()
                continue
            images.append(img)

        images = np.array(images)

        return (np.array(images), np.array(labels))

    from keras.preprocessing.image import img_to_array, load_img

    def load_data(self):
        data_dirname = os.path.join('dataset', '*')
        original_data = []
        for file in glob.glob(data_dirname):
            image = load_img(file).resize((self.img_rows, self.img_cols))
            array = img_to_array(image)
            original_data.append(array)
        data = np.array(original_data) / 255
        print(f'original_data shape: {data.shape}')
        (xtrain, xtest, ytrain, ytest) = train_test_split(data, data, test_size=0.1)
        print(f'xtrain shape: {xtrain.shape}, xtest shape: {xtest.shape}')

        return (
            xtrain,
            xtest
        )

    def get_class_one_hot(self, class_str):
        label_encoded = self.class_names.index(class_str)

        label_hot = np_utils.to_categorical(
            label_encoded, len(self.class_names))
        label_hot = label_hot

        return label_hot

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * self.img_rows//4 * self.img_cols//4,
                        activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((self.img_rows//4, self.img_cols//4, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # train_datagen = ImageDataGenerator(
        #     rescale=1.0 / 255,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     horizontal_flip=True)
        # X_train = train_datagen.flow_from_directory(
        #     '/Users/rasuharu/Dev/gan/image-collector/images/dog',
        #     target_size=(256, 256),
        #     batch_size=batch_size,
        #     class_mode='categorical')

        # X_train, labels = self.load_imgs()

        # X_train = np.load("./gan.p")

        X_train, X_test = self.load_data()

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=8000, batch_size=32, save_interval=50)
