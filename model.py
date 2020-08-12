import pdb
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Sequential,Input,Model
from keras.layers import Dropout,BatchNormalization
from keras import layers
import tensorflow as tf

def build_model(model_name,input_shape):  
    

    model = Sequential()
  
    if model_name == "yolo":
        pass
    elif model_name == "single_card_complete":

        #tfa.layers.InstanceNormalization(axis=3, 
        #                           center=True, 
        #                           scale=True,
        #                           beta_initializer="random_uniform",
        #                           gamma_initializer="random_uniform")

        input_shape = Input(shape=(input_shape),name="Inputs")
        
        Conv7X7 = layers.Conv2D(5,(7, 7),strides=(4,4),activation='relu',padding='same',name="Conv7x7x3")(input_shape)

        Conv1X1 = layers.Conv2D(5,(1, 1),strides=(2,2),activation='relu',padding='same',name="Conv1x1x3")(Conv7X7)
        Conv3X3 = layers.Conv2D(5,(3, 3),strides=(2,2),activation='relu',padding='same',name="Conv3x3x3")(Conv7X7)
        Conv5X5 = layers.Conv2D(5,(5, 5),strides=(2,2),activation='relu',padding='same',name="Conv5x5x3")(Conv7X7)

        concatted = tf.keras.layers.concatenate([Conv1X1, Conv3X3, Conv5X5])

        x = layers.MaxPooling2D((2, 2),name="Maxpool1_4x4")(concatted)
        x = layers.Conv2D(20, (3, 3), activation='relu',name="Conv3x3x5")(x)
        #x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D((2, 2),name="Maxpool2_4x4")(x)
        x = layers.Conv2D(25, (3, 3), activation='relu')(x)
        #x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D((2, 2),name="Maxpool3_2x2")(x)
        x = layers.Conv2D(30, (3, 3), activation='relu',name="Conve3x3x10")(x)
        #x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D((2, 2),name="Maxpool4_2x2")(x)

        flaten = layers.Flatten()(x)
        x = layers.Dense(20, activation='relu')(flaten)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(8, activation='relu')(x)
        #x = layers.Dropout(0.2)(x)

        y = layers.Dense(20, activation='relu')(flaten)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(12, activation='relu')(y)
        #y = layers.Dropout(0.2)(y)

        classe = layers.Dense(6, activation='softmax')(x)
        local  = layers.Dense(4, activation='linear')(y)

        model  = Model(input_shape, [classe, local])
    
        #opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        opt = tf.keras.optimizers.Adam()
        metrics=[['accuracy'],[tf.keras.metrics.RootMeanSquaredError()]]
        model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mean_squared_error'],
            metrics=metrics)
    elif model_name == "single_card_detector":

        input_shape = Input(shape=(input_shape))
        
        Conv7X7 = layers.Conv2D(6,(7, 7),activation='relu',padding='same',name="Firts")(input_shape)

        Conv1X1 = layers.Conv2D(3,(1, 1),activation='relu',padding='same')(Conv7X7)
        Conv3X3 = layers.Conv2D(3,(3, 3),activation='relu',padding='same')(Conv7X7)
        Conv5X5 = layers.Conv2D(3,(5, 5),activation='relu',padding='same')(Conv7X7)

        concatted = tf.keras.layers.concatenate([Conv1X1, Conv3X3, Conv5X5])

        x   = layers.MaxPooling2D((4, 4))(concatted)
        x  = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x   = layers.MaxPooling2D((2, 2))(x)
        x  = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x   = layers.MaxPooling2D((2, 2))(x)
        x  = layers.Conv2D(8, (3, 3), activation='relu',name="SemiFinalConv")(x)
        x   = layers.MaxPooling2D((2, 2),name="SemiFinalPooling")(x)
        x  = layers.Conv2D(16, (3, 3), activation='relu',name="FinalConv")(x)
        x   = layers.MaxPooling2D((4, 4),name="FinalPooling")(x)

        flaten = layers.Flatten()(x)
        y = layers.Dense(16, activation='relu')(flaten)
        y = layers.Dense(16, activation='relu')(y)

        local  = layers.Dense(4, activation='linear')(y)

        model  = Model(input_shape, [local])
    
        model.compile(
                optimizer='adam',
                loss=[ 'mean_squared_error'],
                metrics=[['mse']])
    
    elif model_name == "classifier":
        model.add(layers.Conv2D(8,(5, 5),strides=(2,2),activation='relu',input_shape=input_shape))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(16,(5, 5),strides=(2,2),activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(32,(3, 3),strides=(1,1),activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(32, (3, 3),strides=(1,1),activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3),strides=(1,1),activation='relu'))
        #model.add(layers.MaxPooling2D(4, 4))

        model.add(layers.Flatten())

        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    else:
        raise ValueError("%s is not valid model name"%model_name)

    return model

def build_and_fit_model(args,train):
  input_shape = train.X.shape[1:]
  model = build_model(args.model_name,input_shape)
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir="logs/train.log",histogram_freq=1)]
  history = model.fit(
    train.X,
    train.Y,
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=args.verbose,
    validation_split=0.2,
    workers=4,
    use_multiprocessing=True,
    callbacks=callbacks)
  return model,history
