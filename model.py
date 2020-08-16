import pdb
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import Sequential,Input,Model
from keras.layers import Dropout,BatchNormalization
from keras import layers
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import kerastuner as kt
from tensorboard.plugins.hparams import api as hp

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
        
        x = layers.Conv2D(3,(7, 7),activation='relu',padding='same')(input_shape)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(6,(5, 5),activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(9, (3, 3), activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(12,(3, 3),activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(15, (3, 3), activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(18,(3, 3),activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(21, (3, 3), activation='relu',padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
                
        #x = layers.BatchNormalization(axis=-1)(x)        
        # LayerNorm Layer
        #x =tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True)(x)
        # Groupnorm Layer
        #x = tfa.layers.GroupNormalization(axis=3)(x)

        flatten = layers.Flatten()(x)
        x = layers.Dense(18, activation='relu')(flatten)
        x = layers.Dropout(0.3)(x)

        y = layers.Dense(24, activation='relu')(flatten)
        y = layers.Dropout(0.4)(y)
        y = layers.Dense(12, activation='relu')(y)
        y = layers.Dropout(0.4)(y)

        classe = layers.Dense(6, activation='softmax',name="Class")(x)
        local  = layers.Dense(4, activation='linear',name="Box")(y)

        model  = Model(input_shape, [classe, local])
    
    
        opt = tf.keras.optimizers.Adam(
          learning_rate=0.001,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-07,
          amsgrad=False,
          name="Adam",
        )
        metrics=[['accuracy'],[tf.keras.metrics.RootMeanSquaredError(name="rsme")]]
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


def gen_build_hyper_model(args,train):
  input_shape = train.X.shape[1:]
  if args.model_name == "classifier" :
    def build_hyper_model(hp):

      inputs = tf.keras.Input(shape=input_shape)
      x = inputs
      for i in range(hp.Int('num_conv_layers', 3, 5, default=4)):
        x = layers.Conv2D(
          hp.Int('conv_layers_filter_'+str(i), 3+i, 5+i, default=4+i),
          (3, 3),
          strides=(1,1),
          activation='relu',
          input_shape=input_shape)(x)
        x = layers.MaxPooling2D(2, 2)(x)

      x = layers.Flatten()(x)
      x = layers.Dense(
        units=hp.Int('units',min_value=6,max_value=30,step=3),
        activation='relu')(x)
      x = layers.Dropout(hp.Float('dropout',min_value=0.2,max_value=0.5,step=0.1))(x)
        
      classe = layers.Dense(6, activation='softmax')(x)

      model  = Model(inputs,classe)

      model.compile(
          optimizer=tf.keras.optimizers.Adam(
              hp.Choice('learning_rate',
                        values=[1e-4])),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
      return model
  elif args.model_name == "single_card_detector":
    def build_hyper_model(hp):

      inputs = tf.keras.Input(shape=input_shape)
      x = inputs
      for i in range(hp.Int('num_conv_layers', 3, 5, default=4)):
        x = layers.Conv2D(
          hp.Int('conv_layers_filter_'+str(i), 3+3*i, 10+3*i, default=4+i),
          (3, 3),
          strides=(1,1),
          activation='relu',
          input_shape=input_shape)(x)
        x = layers.MaxPooling2D(2, 2)(x)

      x = layers.Flatten()(x)
      x = layers.Dense(
        units=hp.Int('units',min_value=6,max_value=24,step=6),
        activation='relu')(x)
      x = layers.Dropout(hp.Float('dropout',min_value=0.2,max_value=0.5,step=0.1))(x)
        
      box = layers.Dense(4, activation='linear',name="box")(x)

      model  = Model(inputs,box)

      model.compile(
          optimizer=tf.keras.optimizers.Adam(
              hp.Choice('learning_rate',
                        values=[1e-2,1e-3,1e-4])),
          loss=['mean_squared_error'],
          metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
      return model
  elif args.model_name == "single_card_complete":
    inputs = tf.keras.Input(shape=input_shape)
    def build_hyper_model(hp):
      x = inputs
      for i in range(hp.Int('num_conv_layers', 3, 5, default=4)):
        x = layers.Conv2D(
          hp.Int('conv_layers_filter_'+str(i), 3+i, 5+i, default=4+i),
          (3, 3),
          strides=(1,1),
          activation='relu',
          input_shape=input_shape)(x)
        x = layers.MaxPooling2D(2, 2)(x)

      flatten = layers.Flatten()(x)
      
      y = layers.Dense(
        units=hp.Int('units',min_value=6,max_value=30,step=3),
        activation='relu')(flatten)
      y = layers.Dropout(hp.Float('dropout',min_value=0.2,max_value=0.5,step=0.1))(y)
      
      x = layers.Dense(
        units=hp.Int('units',min_value=6,max_value=30,step=3),
        activation='relu')(flatten)
      x = layers.Dropout(hp.Float('dropout',min_value=0.2,max_value=0.5,step=0.1))(x)
      
      classe = layers.Dense(6, activation='softmax',name="classe")(y)
      box = layers.Dense(4, activation='linear',name="box")(x)

      model  = Model(inputs,[classe,box])

      metrics = metrics=[['accuracy'],[tf.keras.metrics.RootMeanSquaredError(name="rmse")]]
      model.compile(
          optimizer=tf.keras.optimizers.Adam(
              hp.Choice('learning_rate',
                        values=[1e-2,1e-3,1e-4])),
          loss=['categorical_crossentropy', 'mean_squared_error'],
          metrics=metrics)
      return model
  else :
    raise ValueError("args.model_name invalid value '",args.model_name,"'")

  return build_hyper_model

def build_tuner_and_search(args,train):

  builder = gen_build_hyper_model(args,train)

  log_dir="logs/hyperparams.log"
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
  
  if args.model_name == "single_card_complete":
    objective=kt.Objective("val_loss", direction="min")
    callbacks = [
      tensorboard_callback,
      tf.keras.callbacks.EarlyStopping(
        monitor="val_classe_accuracy",
        min_delta=0.05,
        patience=25,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=False),
      tf.keras.callbacks.EarlyStopping(
        monitor="val_box_rmse",
        min_delta=0.01,
        patience=25,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=False)]
  elif args.model_name == "single_card_detector":
    objective=kt.Objective("val_rmse", direction="min")
    callbacks = [
      tensorboard_callback,
      tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse",
        min_delta=0.0025,
        patience=25,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=False)]
  elif args.model_name == "classifier":
    objective='val_accuracy'
    callbacks = [
      tensorboard_callback,
      tf.keras.callbacks.EarlyStopping(
        monitor=objective,
        min_delta=0.05,
        patience=25,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=False)]
  else:
    raise ValueError()

  #tuner = RandomSearch(
  #  builder,
  #  objective=objective,
  #  max_trials=args.max_trials,
  #  executions_per_trial=args.executions_per_trial,
  #  directory='./',
  #  project_name='tune_'+args.model_name,
  #  distribution_strategy=tf.distribute.MirroredStrategy(),
  #  seed=args.seed)

  tuner =  kt.BayesianOptimization(
    builder,
    objective,
    args.max_trials,
    args.num_initial_points,
    args.seed,
    None,
    True,
    True,
    directory='./',
    project_name='tune_'+args.model_name,
    distribution_strategy=tf.distribute.MirroredStrategy())

  tuner.search_space_summary()

  
    #hp.KerasCallback(log_dir, {})]

  tuner.search(
    train.X,
    train.Y,
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=args.verbose,
    validation_split=0.2,
    workers=2,
    use_multiprocessing=True,
    callbacks=callbacks)

  model, = tuner.get_best_models(num_models=1)

  tuner.results_summary()

  return tuner, model

def build_and_fit_model(args,train):
  input_shape = train.X.shape[1:]
  model = build_model(args.model_name,input_shape)
  if args.verbose:
        model.summary()
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="logs/train.log",histogram_freq=1)
  ]
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

def get_dummy():
  def build_hyper_model(hp):
    builder = Model()
    return builder
  return build_hyper_model

def report_tuner(args):

  builder = get_dummy()
  tuner =  kt.BayesianOptimization(
    builder,
    'loss',
    args.max_trials,
    args.num_initial_points,
    args.seed,
    None,
    True,
    True,
    directory='./',
    project_name='tune_'+args.model_name,
    distribution_strategy=tf.distribute.MirroredStrategy())
  tuner.reload()
  tuner.results_summary()
  model, = tuner.get_best_models(num_models=1)
  
  return model


def load_model(args):
    # Set keras verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#'0' if args.verbose else '3'   
    import keras as kr
    model = kr.models.load_model(args.model_name)

    return model
