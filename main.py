import argparse
from dataset import Dataset
import pdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras import layers
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser(description="Main Script for CNN project")

    parser.add_argument(
        "--colab",
        required=False,
        action="store_true")
    parser.add_argument(
        "--data-dir",
        required=False,
        default="./datasets/")
    parser.add_argument(
        "--preprocess",
        required=False,
        default="single_card")
    parser.add_argument(
        "--use_enc",
        required=False,
        action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    label_to_num = {
        "0.0": -1,
        "king": 0,
        "queen": 1,
        "jack": 2,
        "nine": 3,
        "ten": 4,
        "ace": 5
    } if args.use_enc else []

    train = Dataset(args.data_dir, "train", label_to_num,
                    preprocess=args.preprocess)
    test = Dataset(args.data_dir, "test", label_to_num,
                   preprocess=args.preprocess)

    # train.save_yolo_pos_version(2,2)
    # test.save_yolo_pos_version(2,2)
    # train.save_yolo_full_version(2,2)
    # test.save_yolo_full_version(2,2)
    train.get_images()
    test.get_images()
    # print(np.shape(train.X))
    ######################### Keras #########################
    train_x = train.X
    train_y = np.array(train.Y)

    train_xshape = list(train_x.shape)
    train_xshape.append(1)
    train_x = train_x.reshape(train_xshape)

    input_shape = train_x.shape[1:]
    num_class = train_y.shape[1]

    batch_size = 64
    epochs = 5
    k = 3

    model = Sequential()
    model.add(layers.Conv2D(
        32, (k, k), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (k, k), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_class, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=epochs,
                        batch_size=batch_size, verbose=1)

    model.summary()


if("__main__" == __name__):
    main()
