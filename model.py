import pdb
import numpy as np

def build_model(args,train):

    input_shape = train.X.shape[1:]
    print(input_shape)
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras import Sequential,Input,Model
    from keras import layers
    import tensorflow as tf

    model = Sequential()
  
    if args.model_name == "yolo":
        pass
    elif args.model_name == "single_card_complete":

        input_shape = Input(shape=(input_shape))
        
        Conv7X7 = layers.Conv2D(6,(7, 7),activation='relu',padding='same',name="Firts")(input_shape)

        Conv1X1 = layers.Conv2D(3,(1, 1),activation='relu',padding='same')(Conv7X7)
        Conv3X3 = layers.Conv2D(3,(3, 3),activation='relu',padding='same')(Conv7X7)
        Conv5X5 = layers.Conv2D(3,(5, 5),activation='relu',padding='same')(Conv7X7)

        concatted = tf.keras.layers.concatenate([Conv1X1, Conv3X3, Conv5X5])

        x = layers.MaxPooling2D((4, 4))(concatted)
        x = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu',name="SemiFinalConv")(x)
        x = layers.MaxPooling2D((2, 2),name="SemiFinalPooling")(x)
        x = layers.Conv2D(16, (3, 3), activation='relu',name="FinalConv")(x)
        x = layers.MaxPooling2D((4, 4),name="FinalPooling")(x)

        flaten = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(flaten)
        x = layers.Dense(16, activation='relu')(x)

        y = layers.Dense(16, activation='relu')(flaten)
        y = layers.Dense(16, activation='relu')(y)
        

        classe = layers.Dense(6, activation='softmax')(x)
        local  = layers.Dense(4, activation='linear')(y)

        model  = Model(input_shape, [classe, local])
    
        model.compile(
            optimizer='adam',
            loss=['categorical_crossentropy', 'mean_squared_error'],
            metrics=[['accuracy'], ['mse']])
    elif args.model_name == "single_card_detector":

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
    
    elif args.model_name == "classifier":
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
        raise ValueError("%s is not valid model name"%args.model_name)

    history = model.fit(
        train.X,
        train.Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose)

    return model, history
