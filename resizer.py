## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os
import argparse

def valid_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = argparse.ArgumentParser(description="Script to rescale all images in folder, keeping aspect ratio")

parser.add_argument(
    "--scale-factor",
    type=float,
    required=False,
    default=8,
    help="Scaler to use"
)
parser.add_argument(
    "--dir-path",
    type=valid_dir_path,
    required=True,
    help="Path to directory"
)

args = parser.parse_args()

exts = ['JPG','PNG','BMP','jpg','png','bmp']

print("Target dir ",args.dir_path)

scale = 1.0/args.scale_factor
print("Rescaling by ",scale)
for filename in os.listdir(args.dir_path):
    file_path = os.path.join(args.dir_path,filename)
    _,ext = filename.split('.')
    # If the images are not .JPG images, change the line below to match the image type.
    print("file ",filename)
    if ext in exts:
        image = cv2.imread(file_path)
        resized = cv2.resize(image,None,fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print("saving","scaled_"+filename)
        new_file_path = os.path.join(args.dir_path,"scaled_"+filename)
        cv2.imwrite(new_file_path,resized)
