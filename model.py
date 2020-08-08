import pdb
import numpy as np
from keras import Sequential
from keras import layers
import tensorflow as tf

def build_model(args,train):
    
    input_shape = train.X.shape
    model = Sequential()

    if args.model == "yolo":
        input_shape = Input(shape=(input_shape[0], input_shape[1], 1))
        Conv1X1 = layers.Conv2D(
            6,
            (1, 1),
            activation='relu',
            padding='same')(input_shape)
        Conv3X3 = layers.Conv2D(
            8,
            (3, 3),
            activation='relu',
            padding='same')(input_shape)
        Conv5X5 = layers.Conv2D(
            6,
            (5, 5),
            activation='relu',
            padding='same')(input_shape)
        concatted = tf.keras.layers.concatenate([Conv1X1, Conv3X3, Conv5X5])
        Max1   = layers.MaxPooling2D(2, 2)(concatted)
        Conv1  = layers.Conv2D(16, (3, 3), activation='relu')(Max1)
        Max2   = layers.MaxPooling2D(2, 2)(Conv1)
        flaten = layers.Flatten()(Max2)
        final1 = layers.Dense(32, activation='relu')(flaten)
        classe = layers.Dense(6, activation='softmax')(final1)
        local  = layers.Dense(4, activation='linear')(final1)
        model  = Model(input_shape, [classe, local])
    
        model.compile(
                optimizer='adam',
                loss=['categorical_crossentropy', 'mean_squared_error'],
                metrics=[['accuracy'], ['mse']])
    
    elif args.model == "classifier":
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

    else:
        raise ValueError("%s is not valid model name"%args.model)

    history = model.fit(
        train.X,
        train.Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose)

    return model, history
