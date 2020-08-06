#! /usr/bin/python3

from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import pdb
import os

targets = ["test", "train"]
for tgt in targets:
    img_dir = "./datasets/images/"+tgt+"/"
    img_norm_dir = "./datasets/images_norm/"+tgt+"/"
    for img_file in os.listdir(img_dir):
        path = os.path.join(img_dir, img_file)
        base_name, ext = img_file.split('.')
        if path.endswith(".JPG") or path.endswith(".jpg"):
            img = np.array(Image.open(path).convert('L'))
            img_norm = (img-img.mean())/img.std()
            norm_path = os.path.join(img_norm_dir, base_name)
            np.save(norm_path, img_norm),
