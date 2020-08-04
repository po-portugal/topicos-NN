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

    X_train, Y_train = Dataset("./images_norm/","train").get_input_output()
    #X_test, Y_test  = Dataset("./images_norm/","test").get_input_output()
    pdb.set_trace()

if("__main__" == __name__):
    main()

