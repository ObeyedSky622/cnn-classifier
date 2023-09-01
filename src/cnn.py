import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import cv2
from keras.layers import Conv2D, Flatten, Dense, Dropout,MaxPooling2D, Rescaling
from keras.models import Sequential
from keras.utils import image_dataset_from_directory
import pathlib
import matplotlib.pyplot as plt
from keras import layers
from keras.regularizers import L1, L2
import tensorflow as tf
# preprocess


def load_preprocess_gray(path):
    x_data_train = []
    y_data_train = []
    x_data_test = []
    y_data_test = []

    for i in range(50):
        for j in range(10):
            lin_img = cv2.imread(path + 'lin_'+str(i)+'_'+str(j) +
                                 ".png", cv2.IMREAD_GRAYSCALE)
            geo_img = cv2.imread(path + "quad_" + str(i) + '_'+str(j) +
                                 ".png", cv2.IMREAD_GRAYSCALE)

            lin_img_rs = cv2.resize(lin_img, (128, 128))
            geo_img_rs = cv2.resize(geo_img, (128, 128))

            if j >= 7:
                x_data_test.append(lin_img_rs)
                x_data_test.append(geo_img_rs)
                y_data_test.append(0)
                y_data_test.append(1)
            else:

                x_data_train.append(lin_img_rs)
                x_data_train.append(geo_img_rs)

                y_data_train.append(0)
                y_data_train.append(1)

    x_data_train = np.asarray(x_data_train)
    x_data_test = np.asarray(x_data_test)
    y_data_train = np.asarray(y_data_train)
    y_data_test = np.asarray(y_data_test)

    x_data = x_data / 255.0

    x_data = np.reshape(x_data, (1000, 128, 128, 1))
    return x_data_train, x_data_test, y_data_train, y_data_test

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

    # model = Sequential()
    # # add model layers
    # model.add(Conv2D(30, kernel_size=7, activation='relu',
    #           input_shape=(128, 128, 3), padding='same', kernel_regularizer=L1(0.005)))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss="binary_crossentropy",
    #               optimizer="rmsprop", metrics=["accuracy"])
    # model.summary()
    inputs = keras.Input(shape=(128, 128, 3))
    x = Rescaling(1./255)(inputs)
    x = Conv2D(filters=32, kernel_size=7, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    # x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    # x = Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop", metrics=["accuracy"])
    return model


def main():
    # path to img directory

    dataset_path = pathlib.Path("lin_vs_quad")
    
    # train_dataset = image_dataset_from_directory(
    #     dataset_path / "train", image_size=(128, 128), batch_size=16, shuffle=False)

    validation_dataset = image_dataset_from_directory(
        dataset_path / "validation", image_size=(128, 128), batch_size=16)

    # test_dataset = image_dataset_from_directory(
    #     dataset_path / "test", image_size=(128, 128), batch_size=16)

    # model = gen_custom_model()
    # print("Model Generated!")
    iterator = iter(validation_dataset)
    
    first_batch_and_labels  = iterator.get_next()
    
    
    first_batch = first_batch_and_labels[0].numpy()
    
    first_im = first_batch[0, :, :, :]
    
    img_from_dir = cv2.imread("lin_vs_quad/validation/lin/lin_0_5.png")
    img_from_dir = cv2.resize(img_from_dir, (128,128))
    cv2.imshow("",first_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.figure()
    plt.imshow(img_from_dir)
    plt.show()
    print()
    # # train the model
    # print("==============Begin Model Training==================")
    # history = model.fit(train_dataset, validation_data=(
    #     validation_dataset), epochs=10)
    # model.evaluate(test_dataset)

    # accuracy = history.history["accuracy"]
    # val_accuracy = history.history["val_accuracy"]
    # loss = history.history["loss"]
    # val_loss = history.history["val_loss"]
    # epochs = range(1, len(accuracy) + 1)
    # plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    # plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    # plt.title("Training and validation accuracy")
    # plt.legend()
    # plt.figure()
    # plt.plot(epochs, loss, "bo", label="Training loss")
    # plt.plot(epochs, val_loss, "b", label="Validation loss")
    # plt.title("Training and validation loss")
    # plt.legend()
    # plt.show()


main()
