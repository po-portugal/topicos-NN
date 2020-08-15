import pdb
import numpy as np
from dataset import Dataset
from model import build_tuner_and_search
from args import get_args
from main import load_dataset

def main():
    args = get_args()

    train = load_dataset(args)

    model = build_tuner_and_search(args,train)

    model.save(args.model_name)

if("__main__" == __name__):
    main()
