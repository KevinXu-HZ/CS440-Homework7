import argparse
import sys

from viterbi_1 import viterbi_1
import utils

"""
This file contains the main application that is run for this MP.
"""


def main(args):
    print("Loading dataset...")
    train_set = utils.load_dataset(args.training_file)
    test_set = utils.load_dataset(args.test_file)
    test_typos = [item[0] for item in test_set]
    test_goal = [item[1] for item in test_set]
    print("Loaded dataset")
    print()

    print("Running Viterbi 1...")
    test_predictions = viterbi_1(train_set, test_typos)
    cer, correct_charcounter, wrong_charcounter = utils.evaluate_accuracies(test_predictions, test_goal)

    print("Character Error Rate: {:.2f}%".format(cer * 100))
    print("\tTop K Wrong Character Predictions: {}".format(utils.topk_charcounter(wrong_charcounter, k=4)))
    print("\tTop K Correct Character Predictions: {}".format(utils.topk_charcounter(correct_charcounter, k=4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP4 HMM')
    parser.add_argument('--train', dest='training_file', type=str,
                        help='the file of the training data')
    parser.add_argument('--test', dest='test_file', type=str,
                        help='the file of the testing data')
    args = parser.parse_args()
    if args.training_file == None or args.test_file == None:
        sys.exit('You must specify training file and testing file!')

    main(args)