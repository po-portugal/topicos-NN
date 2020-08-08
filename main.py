import pdb
import numpy
from dataset import Dataset
from model import build_model
from args import get_args

def print_model_results(args,train,test,model):
    if args.verbose:
        model.summary()
    score_train = model.evaluate(train.X, train.Y, verbose=args.verbose)
    print('Train loss/accuracy: ', score_train)

    score_test = model.evaluate(test.X, test.Y, verbose=args.verbose)
    print('Test loss/accuracy: ', score_test)

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
    } if args.use_enc else []

    train = Dataset(args.data_dir, "train", label_to_num,
                    preprocess=args.preprocess)
    test = Dataset(args.data_dir, "test", label_to_num,
                   preprocess=args.preprocess)

    train.get_images()
    test.get_images()

    if args.check_dataset:
        np.random.seed(args.seed)
        train.print_check()
        test.print_check()

    model = build_model(args,train)

    model.save(args.model_name)
    if args.print_results:
        print_model_results(args,train,test,model)

if("__main__" == __name__):
    main()
