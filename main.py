#! /usr/bin/python3

import argparse
from dataset import Dataset
import pdb

def get_args():
    parser = argparse.ArgumentParser(description="Main Script for CNN project")

    parser.add_argument(
        "--colab",
        required=False,
        action="store_true")
    parser.add_argument(
        "--image-dir",
        required=False,
        default="./images_norm/")
    parser.add_argument(
        "--preprocess",
        required=False,
        default="basic")

    return parser.parse_args()

def main():
    args = get_args()

    #X_train, Y_train = Dataset(args.image_dir,"train",preprocess=args.preprocess).get_input_output()
    #X_test,  Y_test  = Dataset(args.image_dir,"test", preprocess=args.preprocess).get_input_output()

    train = Dataset(args.image_dir,"train",preprocess=args.preprocess)
    test  = Dataset(args.image_dir,"test", preprocess=args.preprocess)

    train.save_single_card_dataset()
    test.save_single_card_dataset()

    pdb.set_trace()

if("__main__" == __name__):
    main()

