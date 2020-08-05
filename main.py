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
        "--data-dir",
        required=False,
        default="./datasets/")
    parser.add_argument(
        "--preprocess",
        required=False,
        default="single_card")

    return parser.parse_args()

def main():
    args = get_args()

    train = Dataset(args.data_dir,"train",preprocess=args.preprocess)
    test  = Dataset(args.data_dir,"test", preprocess=args.preprocess)

    train.get_images()
    pdb.set_trace()

if("__main__" == __name__):
    main()

