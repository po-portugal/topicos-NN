import pdb
import numpy as np
from dataset import Dataset
from model import report_tuner
from args import get_args
from main import load_dataset

def main():
  args = get_args()

  model = report_tuner(args)

  model.save(args.model_name)

if("__main__" == __name__):
    main()
