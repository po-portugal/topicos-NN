import argparse
from dataset import Dataset
import pdb
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras import Input, Model, Sequential
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
        "nine": 0,
        "ten": 1,
        "jack": 2,
        "queen": 3,
        "king": 4,
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

    test_x = test.X
    test_y = np.array(test.Y)

    train_xshape = list(train_x.shape)
    train_xshape.append(1)
    train_x = train_x.reshape(train_xshape)

    test_xshape = list(test_x.shape)
    test_xshape.append(1)
    test_x = test_x.reshape(test_xshape)

    input_shape = train_x.shape[1:]
    num_class = train_y.shape[:]
    #import ipdb
    # ipdb.set_trace()

    batch_size = 32
    epochs = 7
    k = 3
    model = Sequential()

    input_shape = Input(shape=(input_shape[0], input_shape[1], 1))
    Conv1X1 = layers.Conv2D(6, (1, 1), activation='relu',
                            padding='same')(input_shape)
    Conv3X3 = layers.Conv2D(8, (3, 3), activation='relu',
                            padding='same')(input_shape)
    Conv5X5 = layers.Conv2D(6, (5, 5), activation='relu',
                            padding='same')(input_shape)
    concatted = tf.keras.layers.concatenate([Conv1X1, Conv3X3, Conv5X5])
    Max1 = layers.MaxPooling2D(2, 2)(concatted)
    Conv1 = layers.Conv2D(16, (k, k), activation='relu')(Max1)
    Max2 = layers.MaxPooling2D(2, 2)(Conv1)
    flaten = layers.Flatten()(Max2)
    final1 = layers.Dense(32, activation='relu')(flaten)
    classe = layers.Dense(6, activation='softmax')(final1)
    local = layers.Dense(4, activation='linear')(final1)
    Saida = tf.keras.layers.concatenate([classe, local])
    model = Model(input_shape, Saida)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y[:], epochs=epochs,
                        batch_size=batch_size, verbose=1)

    model.summary()

    model.save("TiagoVr1.h5")

    socore_train = model.evaluate(train_x, train_y[:], verbose=0)
    print('Train loss: ', socore_train[0])
    print('Train acurracy: ', socore_train[1])

    socore_validation = model.evaluate(test_x, test_y[:], verbose=0)
    print('Test loss: ', socore_validation[0])
    print('Test acurracy: ', socore_validation[1])
    dot_img_file = 'model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


if("__main__" == __name__):
    main()
