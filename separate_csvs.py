import argparse
from dataset import Dataset
import pdb
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Split original csv into other types")

    parser.add_argument(
        "--data-dir",
        required=False,
        default="./datasets/")

    return parser.parse_args()

def scaled_csv(args):
    train = Dataset(
        args.data_dir,
        "train",
        {},
        preprocess="original")
    test = Dataset(
        args.data_dir,
        "test",
        {},
        preprocess="original")

    train.save_scaled_version()
    test.save_scaled_version()


def single_card_csv(args):
    train = Dataset(
        args.data_dir,
        "train",
        {},
        preprocess="scaled")
    test = Dataset(
        args.data_dir,
        "test",
        {},
        preprocess="scaled")

    train.save_single_card_dataset()
    test.save_single_card_dataset()


def yolo_pos_csv(args,sizes):
    train = Dataset(
        args.data_dir,
        "train",
        {},
        preprocess="scaled")
    test = Dataset(
        args.data_dir,
        "test",
        {},
        preprocess="scaled")

    train.save_yolo_pos_version(*sizes)
    test.save_yolo_pos_version(*sizes)

def yolo_csv(args,label_to_num,sizes):
    train = Dataset(
        args.data_dir,
        "train",
        label_to_num,
        preprocess="scaled")
    test = Dataset(
        args.data_dir,
        "test",
        label_to_num,
        preprocess="scaled")

    train.save_yolo_full_version(*sizes)
    test.save_yolo_full_version(*sizes)

def single_card_center_csv(args):
    train = Dataset(
        args.data_dir,
        "train",
        {},
        preprocess="original")
    test = Dataset(
        args.data_dir,
        "test",
        {},
        preprocess="original")

    train.save_single_card_center_dataset()
    test.save_single_card_center_dataset()


def main():
    args = get_args()

    label_to_num = {
        "0.0": -1,
        "nine": 0,
        "ten": 1,
        "jack": 2,
        "queen": 3,
        "king": 4,
        "ace": 5
    }

    print("Start...")
    single_card_center_csv(args)
    scaled_csv(args)
    single_card_csv(args)
    yolo_pos_csv(args,(2,2))
    yolo_csv(args,label_to_num,(2,2))
    print("...Done")

if("__main__" == __name__):
    main()
