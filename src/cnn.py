import keras
import numpy as np
import cv2
import chirp
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.applications import resnet50
from sklearn.model_selection import train_test_split


# preprocess


def load_preprocess_gray(path):
    x_data = []
    y_data = []

    for i in range(50):
        for j in range(10):
            lin_img = cv2.imread(path + 'lin_'+str(i)+'_'+str(j) +
                                 ".png", cv2.IMREAD_GRAYSCALE)
            geo_img = cv2.imread(path + "quad_" + str(i) + '_'+str(j) +
                                 ".png", cv2.IMREAD_GRAYSCALE)

            lin_img_rs = cv2.resize(lin_img, (128, 128))
            geo_img_rs = cv2.resize(geo_img, (128, 128))

            x_data.append(lin_img_rs)
            x_data.append(geo_img_rs)

            y_data.append(0)
            y_data.append(1)
    y_data = np.asarray(y_data)
    x_data = np.asarray(x_data)

    print("Initial Shape: ", x_data.shape)

    x_data = x_data / 255.0

    x_data = np.reshape(x_data, (1000, 128, 128, 1))
    print("Reshaped Shape: ", x_data.shape)
    return x_data, y_data

# preprocess


def load_preprocess_color(path):
    x_data = []
    y_data = []
    for i in range(50):
        for j in range(10):
            lin_img = cv2.imread(path + 'lin_'+str(i)+'_'+str(j) +
                                 ".png")
            geo_img = cv2.imread(path + "quad_" + str(i) +
                                 ".png")

            lin_img_rs = cv2.resize(lin_img, (128, 128))
            geo_img_rs = cv2.resize(geo_img, (128, 128))

            x_data.append(lin_img_rs)
            x_data.append(geo_img_rs)

            y_data.append(0)
            y_data.append(1)

    y_data = np.asarray(y_data)
    x_data = np.asarray(x_data)

    print("Initial Shape: ", x_data.shape)

    x_data = x_data / 255.0

    x_data = np.reshape(x_data, (1000, 128, 128, 1))
    print("Reshaped Shape: ", x_data.shape)
    return x_data, y_data


def gen_custom_model():

    model = Sequential()
    # add model layers
    model.add(Conv2D(30, kernel_size=7, activation='relu',
              input_shape=(128, 128, 1), padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(60, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def main():
    # path to img directory
    path = "data/"

    # load and preprocess (normalize) the images
    x_data, y_data = load_preprocess_gray(path)

    print("Images loaded!")
    # generate the model to use
    print("Generating model... ...")
    model = gen_custom_model()
    print("Model Generated!")

    # split into train and testing groups
    print("Splitting data... ...")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    print(f"Training on {len(y_train)} samples")
    print(f"Testing on {len(y_test)} samples")
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    # train the model
    print("==============Begin Model Training==================")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    model.evaluate(x_test, y_test)


main()
