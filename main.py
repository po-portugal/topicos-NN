#! /usr/bin/python3

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Main Script for CNN project")

    parser.add_argument(
        "--colab",
        required=False,
        action="store_true")

    return parser.parse_args()

def main():
    args = get_args()

    print(args)

if("__main__" == __name__):
    main()

