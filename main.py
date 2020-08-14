import pdb
import numpy as np
from dataset import Dataset
from model import build_model,build_and_fit_model
from args import get_args

def load_dataset(args):
    label_to_num = {
        "0.0": -1,
        "nine": 0,
        "ten": 1,
        "jack": 2,
        "queen": 3,
        "king": 4,
        "ace": 5
    } if args.use_enc else []

    train = Dataset(args.data_dir, "train", label_to_num,
                    preprocess=args.preprocess)

    train.set_labels_slice(args.model_name)

    train.get_images()

    if args.check_dataset:
        np.random.seed(args.seed)
        train.print_check(args.model_name)

    return train

def main():
    args = get_args()

    train = load_dataset(args)

    model,history = build_and_fit_model(args,train)

    model.save(args.model_name)

if("__main__" == __name__):
    main()
