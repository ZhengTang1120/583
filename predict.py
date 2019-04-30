from model_dy import *
from utils import *
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    classifier = LSTMClassifier.load(args.model)

    lang, train_set = prepare_jldata("snli_1.0/snli_1.0_train.jsonl")
    lang, test_set = prepare_jldata("snli_1.0/snli_1.0_test.jsonl", lang)

    gls = list()
    pls = list()
    for datapoint in test_set:
        premise = datapoint["premise"]
        hypothesis = datapoint["hypothesis"]
        gold_label = datapoint["gold_label"]
        gls.append(gold_label)
        predict = classifier.get_relation(premise, hypothesis)
        pls.append(predict)
    res = np.array([pls[i]==gls[i] for i in range(len(pls))])
    acc = float(np.sum(res)/len(gls))
    print ("Accuracy: %2f"%acc)
