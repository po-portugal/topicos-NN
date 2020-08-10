def test_model():

  import numpy as np
  from args import get_args
  from PIL import Image 
  import matplotlib.pyplot as plt
  
  import matplotlib.patches as patches
  import os
  import pdb



  def get_image_paths(args):
      np.random.seed(args.seed)
      img_dir = args.data_dir+"images/test/"
      file_list = os.listdir(img_dir)
      img_file_list = [f for f in file_list if not(f.endswith('.xml'))]
      index = np.random.randint(len(img_file_list))
      img_path = os.path.join(img_dir,img_file_list[index])

      img_dir = args.data_dir+"images_norm/test/"
      base_name, _ = img_file_list[index].split('.')
      img_norm_path = os.path.join(img_dir,base_name+".npy")

      return img_path, img_norm_path

  def post_process(args,prediction):
      labels = ["ace","king","queen","jack","ten","nine"]
      
      if args.model_name == "yolo":
        pred_pos = prediction
        #max_index = np.argmax(pred_labels)
        #pred_label = labels[max_index]
        #pred_conf  = pred_labels[0,max_index]
        pred_pos = pred_pos*np.array([378,504,378,504])
        pred_pos = [int(pos) for pos in pred_pos[0]]
      if args.model_name == "classifier":
        max_index  = np.argmax(prediction)
        pred_label = labels[max_index]
        pred_conf  = prediction[0,max_index] 
        pred_pos = None
      
      return pred_pos

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

  def print_results(args,img_path,test,model):
    prediction = model.predict(test) # shape required (1,504,378,1)
    pred_pos = post_process(args,prediction)
    
    print("Test image : '",img_path,"'")
    plt.imshow(np.asarray(Image.open(img_path)))
    if pred_pos is not None :
      draw_bound_box(pred_label,pred_pos)
    plt.show()
  
    print("Predicted Label '",pred_label,"' with '",pred_conf,"' confidence")
    print("Predicted pos '",pred_pos,"'")

    if args.verbose:
      print("Predicted: ",prediction) 
      model.summary()
      print("Model input ",test)

  args = get_args()

  img_path, img_norm_path = get_image_paths(args)

  test = np.load(img_norm_path)
  test = test.reshape([1]+list(test.shape)+[1])

  model = load_model(args)

  print_results(args,img_path,test,model)

pred_label = 0
pred_conf = 0

test_model()
