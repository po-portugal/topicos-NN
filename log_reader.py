import pdb
import numpy as np
from dataset import Dataset
from model import report_tuner
from args import get_args
from main import load_dataset

def main():
  args = get_args()

  train = load_dataset(args)

  model = report_tuner(args,train)

  model.save(args.model_name,train)

if("__main__" == __name__):
    main()
