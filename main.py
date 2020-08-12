import pdb
import numpy
import numpy as np
from dataset import Dataset
from model import build_model
from args import get_args

def print_model_results(args,train,test,model):
    if args.verbose:
        model.summary()

    import tensorflow as tf
    dot_img_file = args.model_name+'_model.png'
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=True)
    score_train = model.evaluate(train.X, train.Y, verbose=args.verbose)
    print('Train loss/accuracy: ', score_train)

    score_test = model.evaluate(test.X, test.Y, verbose=args.verbose)
    print('Test loss/accuracy: ', score_test)

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
    test = Dataset(args.data_dir, "test", label_to_num,
                   preprocess=args.preprocess)

    train.set_labels_slice(args.model_name)
    test.set_labels_slice(args.model_name)

    train.get_images()
    test.get_images()

    if args.check_dataset:
        np.random.seed(args.seed)
        train.print_check(args.model_name)
        test.print_check(args.model_name)

    return train, test

def main():
    args = get_args()

    train, test = load_dataset(args)

    input_shape = train.X.shape[1:]
    model = build_model(args.model_name,input_shape)
    history = model.fit(
        train.X,
        train.Y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        validation_split=0.2,
        workers=4,
        use_multiprocessing=True)

    model.save(args.model_name)
    if args.print_results:
        print_model_results(args,train,test,model)

if("__main__" == __name__):
    main()
