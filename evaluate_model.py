def evaluate_model():

  import numpy as np
  from args import get_args
  from dataset import Dataset
  from model import load_model
  
  import os
  import pdb

  def print_model_results(args,data_set,model):
  
    import tensorflow as tf
    dot_img_file = args.model_name+'_model.png'
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=True)

    score = model.evaluate(data_set.X, data_set.Y, verbose=args.verbose)
    print(args.set_dir,' loss/accuracy: ', score)

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

  data_set = Dataset(args.data_dir, args.set_dir, label_to_num,
                    preprocess=args.preprocess)

  data_set.set_labels_slice(args.model_name)
  data_set.get_images()

  model = load_model(args)

  print_model_results(args,data_set,model)

evaluate_model()