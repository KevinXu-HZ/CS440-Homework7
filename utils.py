import collections
import jiwer

START_TAG = "^"
END_TAG = "$"


def evaluate_accuracies(predicted_words, true_words):
    """
    :param predicted_sentences:
    :param true_words:
    :return: (Accuracy, correct word-tag counter, wrong word-tag counter)
    """
    assert len(predicted_words) == len(true_words), "The number of predicted words {} does not match the true number {}".format(len(predicted_words), len(true_words))

    correct_charcounter = {}
    wrong_charcounter = {}
    correct = 0
    wrong = 0
    clean_preds = []
    clean_gold = []
    for pred_word, true_word in zip(predicted_words, true_words):
        clean_pred = pred_word.replace("^", "").replace("$", "")
        clean_true = true_word.replace("^", "").replace("$", "")
        clean_preds.append(clean_pred)
        clean_gold.append(clean_true)
        assert len(clean_pred) == len(clean_true), "The predicted word length {} does not match the true length {}".format(len(pred_word), len(true_word))
        for pred_char, true_char in zip(clean_pred, clean_true):
                if true_char in [START_TAG, END_TAG]:
                    continue
                if pred_char == true_char:
                    if pred_char not in correct_charcounter:
                        correct_charcounter[pred_char] = collections.Counter()
                    correct_charcounter[pred_char].update({true_char: 1})
                    correct += 1
                else:
                    if pred_char not in wrong_charcounter:
                        wrong_charcounter[pred_char] = collections.Counter()
                    wrong_charcounter[pred_char].update({true_char: 1})
                    wrong += 1

    output = jiwer.process_characters(clean_gold, clean_preds)
    return output.cer, correct_charcounter, wrong_charcounter


def topk_charcounter(charcounter, k):
    top_items = sorted(charcounter.items(), key=lambda item: sum(item[1].values()), reverse=True)[:k]
    top_items = list(map(lambda item: (item[0], dict(item[1])), top_items))
    return top_items


def load_dataset(data_file):
    if not data_file.endswith(".txt"):
        raise ValueError("File must be a .txt file")

    pairs = []
    with open(data_file, 'r', encoding='UTF-8') as f:
        # next(f)
        for line in f:
            pair = line.split(',')
            pairs.append((START_TAG+pair[0].strip()+END_TAG, START_TAG+pair[1].strip()+END_TAG))
    return pairs