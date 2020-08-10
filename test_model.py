import numpy as np
from args import get_args
from PIL import Image 
import matplotlib.pyplot as plt
import os

args = get_args()

# Set keras verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#'0' if args.verbose else '3'   
import keras as kr

model = kr.models.load_model(args.model_name)

np.random.seed(args.seed)
img_dir = args.data_dir+"images/test/"
file_list = os.listdir(img_dir)
img_file_list = [f for f in file_list if not(f.endswith('.xml'))]
index = np.random.randint(len(img_file_list))
img_path = os.path.join(img_dir,img_file_list[index])

img_dir = args.data_dir+"images_norm/test/"
base_name, _ = img_file_list[index].split('.')
img_norm_path = os.path.join(img_dir,base_name+".npy")
test=np.load(img_norm_path)
test = test.reshape([1]+list(test.shape)+[1])

print("Test image : '",img_path,"'")
plt.imshow(np.asarray(Image.open(img_path)))
plt.show()
print("Predicted: ",model.predict(test)) # shape required (1,504,378,1)
if args.verbose:
  model.summary()
  print("Model input ",test)
