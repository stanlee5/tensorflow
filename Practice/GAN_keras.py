
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu1, floatX=float32, lib.cnmem=0.2"
os.environ["KERAS_BACKEND"] = "theano"

from keras.models import Sequential
from keras.layers import BatchNormalization, Reshape
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import adam, SGD

from keras.datasets import mnist

from PIL import Image

import numpy as np
import math, argparse


# for MNIST Case

def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
        64, 5, 5,
        border_mode='same',
        input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))

    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((128, 7, 7), input_shape=(128*7*7, )))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

    return model

def generator_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def get_break():
    a = raw_input()

def merge_image(image):
    n = image.shape[0]
    size = image.shape[2:]
    w = int(math.sqrt(n))
    h = int(math.ceil(float(n)/w))
    result = np.zeros((h*size[0], w*size[1]), dtype=image.dtype)

    for idx, img in enumerate(image):
        i = int(idx / w)
        j = idx % w
        result[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]] = img[0, :, :]
    return result


def train(batch_size=128, epoch_size=30, noise_dim=100,
          save_path="kerasVer/",
          lr_G=0.0005, lr_D=0.0005, lr_GD=0.0005,
          D_epoch=1):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape((x_train.shape[0], 1) + x_train.shape[1:])
    D = discriminator_model()
    G = generator_model()
    GD = generator_discriminator(G, D)

    g_opt = adam(lr=lr_G)
    G.compile(loss='binary_crossentropy', optimizer=g_opt)
    #G.summary()

    gd_opt = adam(lr=lr_D)
    GD.compile(loss='binary_crossentropy', optimizer=gd_opt)
    #GD.summary()

    d_opt = adam(lr=lr_GD)
    D.trainable = True
    D.compile(loss='binary_crossentropy', optimizer=d_opt)
    #D.summary()

    print("Batch size: {}".format(batch_size))
    print("Noise dimension: {}".format(noise_dim))
    print("Learning rate")
    print("- generator: {}".format(lr_G))
    print("- discriminator: {}".format(lr_D))
    print("- G-D connection: {}".format(lr_GD))
    print("D-epoch: {}".format(D_epoch))
    print("Save path: {}".format(save_path))

    noise = np.zeros((batch_size, noise_dim))
    for epoch in xrange(epoch_size):
        print("Epoch: {}".format(epoch))

        for batch in xrange(int(x_train.shape[0]/batch_size)):
            for i in xrange(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, noise_dim)
            real_img = x_train[batch * batch_size : (batch+1)*batch_size]
            fake_img = G.predict(noise, verbose=0)

            x = np.concatenate((real_img, fake_img))
            y = np.asarray([1] * batch_size + [0] * batch_size)

            D.trainable = True
            for depoch in xrange(D_epoch):
                d_loss = D.train_on_batch(x,y)
            print ("-batch {}: d_loss : {}".format(batch, d_loss))

            for i in xrange(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, noise_dim)
            D.trainable = False
            g_loss = GD.train_on_batch(noise, [1]*batch_size)
            print("-batch {}: g_loss : {}".format(batch, g_loss))

            # Write generate images
            if (epoch < 10 and batch%10==0) or batch%50==0:
                image = merge_image(fake_img)
                image = image*127.5 + 127.5
                # convert into 0~255
                Image.fromarray(image.astype(np.uint8)).save(
                    save_path+str(epoch)+"-"+str(batch)+".png")

        G.save_weights('kerasVer/g_weight', overwrite=True)
        D.save_weights('kerasVer/d_weight', overwrite=True)

def generate(batch_size=128,
             noise_dim=100):
    G = generator_model()
    g_opt = adam(lr=0.0002)
    G.compile(loss='binary_crossentropy', optimizer=g_opt)
    G.load_weights('kerasVer/g_weight')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="samples/")
    parser.add_argument("--batch", type=int, default=256)

    #parser.set_defaults(path="samples/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    save_path = args.path
    train(batch_size=256, epoch_size=10, save_path=save_path,
          lr_G=0.0005, lr_D=0.0005, lr_GD=0.0005, D_epoch=5)
