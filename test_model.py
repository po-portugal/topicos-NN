def test_model():

  import numpy as np
  from args import get_args
  from PIL import Image 
  import matplotlib.pyplot as plt
  from dataset import Dataset
  from model import load_model
  
  import matplotlib.patches as patches
  import os
  import pdb

  def sort(args,data_sete,predictions):
    if args.model_name == "single_card_complete":
      data_set.Y_rand_arr = np.array([y[1:] for y in data_set.Y_rand])
      #rmse = np.sqrt(np.sum((data_set.Y_rand[:,6:] - predictions[0][:,6:])**2,axis=-1))
      rmse = np.sqrt(np.sum((data_set.Y_rand_arr - predictions[1][:,:6])**2,axis=-1))
      criteria_l = rmse
    elif args.model_name == "single_card_detector":
      data_set.Y_rand_arr = np.array([y[1:] for y in data_set.Y_rand])
      rmse = np.sqrt(np.sum((data_set.Y_rand_arr - predictions[:,:6])**2,axis=-1))
      criteria_l = rmse
    elif args.model_name == "classifier":
      pass
    else:
      raise ValueError("args.sort not recognized '",args.sort,"'")

    if not(args.ascending) :
      criteria_l = -criteria_l

    index = np.argsort(criteria_l)
    criteria_l = criteria_l[index]
    predictions = predictions[index]
    #predictions[0] = predictions[0][index]
    #predictions[1] = predictions[1][index]
    data_set.X_meta_rand  = [data_set.X_meta_rand[i] for i in index]
    data_set.Y_rand  = [data_set.Y_rand[i] for i in index]

    return data_set, predictions, criteria_l

  def draw_bound_box(label,coord,box_color='m'):
    x_ul, y_ul, x_lr, y_lr = coord
    width_box, height_box  = x_lr-x_ul, y_lr-y_ul
    width_label, height_label  = 70, -20
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

  def print_results(args,predictions,data_set,criteria_l):
    for prediction, img_meta,img_label,criteria in zip(predictions,data_set.X_meta_rand,data_set.Y_rand,criteria_l):
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
      print("Sorting criteria '",criteria,"'")


  args = get_args()

  data_set = Dataset(args.data_dir, args.set_dir, {},
          preprocess=args.preprocess)
  data_set.load_rand_images(args.seed,args.num_files)
  data_set.get_post_processor(args.model_name)

  model = load_model(args)

  predictions = model.predict(data_set.X_rand) # shape required (1,504,378,1)
  
  if args.sort:
    data_set, predictions, criteria_l = sort(args,data_set,predictions)
  else:
    criteria_l = [None]*len(data_set.X_meta_rand)
  
  predictions_post = data_set.post_process(predictions)

  if args.verbose:
      model.summary()
  print_results(args,predictions_post,data_set,criteria_l)

test_model()
