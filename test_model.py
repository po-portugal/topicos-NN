import numpy as np
from args import get_args
import os

args = get_args()

# Set keras verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#'0' if args.verbose else '3'   
import keras as kr

model = kr.models.load_model(args.model_name)

if args.verbose:
  model.summary()

np.random.seed(args.seed)
img_dir = args.data_dir+"images_norm/test/"
img_file_list = os.listdir(img_dir)
index = np.random.randint(len(img_file_list))
img_path = os.path.join(img_dir,img_file_list[index])
test=np.load(img_path)
test = test.reshape([1]+list(test.shape)+[1])

print("Test image : '",img_path,"'")
print("Predicted: ",model.predict(test)) # shape required (1,504,378,1)
print("Model input ",test)