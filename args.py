import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Main Script for CNN project")

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="Tiago.h5")
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=10)
    parser.add_argument(
        "--data-dir",
        required=False,
        default="./datasets/")
    parser.add_argument(
        "--preprocess",
        required=False,
        default="single_card")
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true")
    parser.add_argument(
        "--print_results",
        required=False,
        action="store_true")
    parser.add_argument(
        "--use_enc",
        required=False,
        action="store_true")
    parser.add_argument(
        "--check_dataset",
        required=False,
        action="store_true")

    return parser.parse_args()


