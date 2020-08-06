#! /usr/bin/python3

import argparse
from dataset import Dataset
import pdb
import numpy as np


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
    parser.add_argument(
        "--use_enc",
        required=False,
        action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    label_to_num = {
        "0.0": -1,
        "king": 0,
        "queen": 1,
        "jack": 2,
        "nine": 3,
        "ten": 4,
        "ace": 5
    } if args.use_enc else []

    train = Dataset(args.data_dir, "train", label_to_num,
                    preprocess=args.preprocess)
    test = Dataset(args.data_dir, "test", label_to_num,
                   preprocess=args.preprocess)
    #np.savetxt('trainX.csv', train.X, delimiter=',')
    #np.savetxt('testX.csv', test.X, delimiter=',')
    # print(train.x[1])

    # train.save_yolo_pos_version(2,2)
    # test.save_yolo_pos_version(2,2)
    # train.save_yolo_full_version(2,2)
    # test.save_yolo_full_version(2,2)
    # train.get_images()
    # print(train.Y[0])
    return train, test


if("__main__" == __name__):
    main()
