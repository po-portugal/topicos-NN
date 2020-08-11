def test_model():

  import numpy as np
  from args import get_args
  from PIL import Image 
  import matplotlib.pyplot as plt
  from dataset import Dataset
  
  import matplotlib.patches as patches
  import os
  import pdb

  def draw_bound_box(label,coord,box_color='m'):
    x_ul, y_ul, x_lr, y_lr = coord
    width_box, height_box  = x_lr-x_ul, y_lr-y_ul
    width_label, height_label  = 60, -20
    margin = 3
    label_color = 'w'

    # Create a Rectangle patch
    box = patches.Rectangle((x_ul,y_ul),width_box,height_box,linewidth=3,edgecolor=box_color,facecolor='none')
    label_box = patches.Rectangle((x_ul,y_ul),width_label,height_label,linewidth=3,edgecolor=box_color,facecolor=box_color)

    # Add the patch to the Axes
    ax = plt.gcf().gca()
    ax.add_patch(box)
    ax.add_patch(label_box)
    plt.text(x_ul, y_ul-margin,label,color=label_color)

  def load_model(args):
    # Set keras verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#'0' if args.verbose else '3'   
    import keras as kr
    model = kr.models.load_model(args.model_name)

    return model

  def print_results(args,predictions,data_set):
    for prediction, img_meta,img_label in zip(predictions,data_set.X_meta_rand,data_set.Y_rand):
      img_path = "/".join([data_set.dataset_dir,"images",data_set.target,img_meta[0]])
      pred_label, pred_conf, pred_pos = prediction
      plt.imshow(np.asarray(Image.open(img_path)))
      if pred_pos is not None :
        draw_bound_box(pred_label,pred_pos)
      plt.show()
    
      print(args.set_dir+" image : '",img_path,"'")
      print("Ans '",img_label)
      print("Predicted Label '",pred_label,"' with '",pred_conf,"' confidence")
      print("Predicted pos '",pred_pos,"'")


  args = get_args()

  data_set = Dataset(args.data_dir, args.set_dir, {},
          preprocess=args.preprocess)
  data_set.load_rand_images(args.seed,args.num_files)
  data_set.get_post_processor(args.model_name)

  model = load_model(args)

  predictions = model.predict(data_set.X_rand) # shape required (1,504,378,1)
  predictions = data_set.post_process(predictions)

  if args.verbose:
      model.summary()
  print_results(args,predictions,data_set)

test_model()
