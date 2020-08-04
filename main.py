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

    return parser.parse_args()

def main():
    args = get_args()

    train = Dataset("./images_norm/","train")
    test  = Dataset("./images_norm/","test")
    pdb.set_trace()

if("__main__" == __name__):
    main()

