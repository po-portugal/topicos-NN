import pdb
import numpy as np
from keras import Input, Model, Sequential
from keras import layers
import tensorflow as tf


def build_model(args, train):

    input_shape = train.X.shape[1:]
    model = Sequential()

    if args.model_name == "yolo":
        train_x = train.X
        train_y = np.array(train.Y)
        train_xshape = list(train_x.shape)
        train_xshape.append(1)
        train_x = train_x.reshape(train_xshape)

        input_shape = train_x.shape[1:]
        num_class = train_y.shape[:]
        input_shape = Input(shape=(input_shape[0], input_shape[1], 1))
        ##################-------------#####################
        Conv7X7X64_1 = layers.Conv2D(64, (7, 7), strides=(2, 2), activation='relu',
                                     padding='same')(input_shape)
        Max_1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(Conv7X7X64_1)
        ##################-------------#####################
        Conv3X3X192_2 = layers.Conv2D(192, (3, 3), strides=(1, 1), activation='relu',
                                      padding='same')(Max_1)
        Max_2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(Conv3X3X192_2)
        ##################-------------#####################
        Conv_1_1X1_3 = layers.Conv2D(128, (1, 1), activation='relu',
                                     padding='same')(Max_2)
        Conv_1_3X3_3 = layers.Conv2D(256, (3, 3), activation='relu',
                                     padding='same')(Max_2)
        Conv_2_1X1_3 = layers.Conv2D(256, (1, 1), activation='relu',
                                     padding='same')(Max_2)
        Conv_2_3X3_3 = layers.Conv2D(512, (3, 3), activation='relu',
                                     padding='same')(Max_2)
        concatted_1 = tf.keras.layers.concatenate(
            [Conv_1_1X1_3, Conv_1_3X3_3, Conv_2_1X1_3, Conv_2_3X3_3])
        Max_3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(concatted_1)
        ##################-------------#####################
        Conv_1_1X1_4 = layers.Conv2D(256, (1, 1), activation='relu',
                                     padding='same')(Max_3)
        Conv_1_3X3_4 = layers.Conv2D(512, (3, 3), activation='relu',
                                     padding='same')(Max_3)
        Conv_2_1X1_4 = layers.Conv2D(512, (1, 1), activation='relu',
                                     padding='same')(Max_3)
        Conv_2_3X3_4 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(Max_3)
        concatted_2 = tf.keras.layers.concatenate(
            [Conv_1_1X1_4, Conv_1_3X3_4, Conv_2_1X1_4, Conv_2_3X3_4])
        Max_4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(concatted_2)
        ##################-------------#####################
        Conv_1_1X1_5 = layers.Conv2D(512, (1, 1), activation='relu',
                                     padding='same')(Max_4)
        Conv_1_3X3_5 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(Max_4)
        Conv_2_3X3_5 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(Max_4)
        Conv_3_3X3_5 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(Max_4)
        concatted_3 = tf.keras.layers.concatenate(
            [Conv_1_1X1_5, Conv_1_3X3_5, Conv_2_3X3_5, Conv_3_3X3_5])
        ##################-------------#####################
        Conv_1_3X3_6 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(concatted_3)
        Conv_2_3X3_6 = layers.Conv2D(1024, (3, 3), activation='relu',
                                     padding='same')(concatted_3)
        concatted_4 = tf.keras.layers.concatenate([Conv_1_3X3_6, Conv_2_3X3_6])
        ##################-------------#####################
        flaten = layers.Flatten()(concatted_4)
        final1 = layers.Dense(30, activation='relu')(flaten)
        ##################-------------#####################
        classe = layers.Dense(6, activation='softmax')(final1)
        local = layers.Dense(4, activation='linear')(final1)
        ##################-------------#####################

        model = Model(input_shape, [classe, local])

        model.compile(
            optimizer='adam',
            loss=['categorical_crossentropy', 'mean_squared_error'],
            metrics=[['accuracy'], ['mse']])

        history = model.fit(train_x, [train_y[:, :6], train_y[:, 6:]], epochs=args.epochs,
                            batch_size=args.batch_size, verbose=1)

        dot_img_file = 'model_1.png'
        tf.keras.utils.plot_model(
            model, to_file=dot_img_file, show_shapes=True)

    elif args.model_name == "classifier":
        model.add(layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        history = model.fit(
            train.X,
            train.Y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose)

    else:
        raise ValueError("%s is not valid model name" % args.model_name)

    return model, history
