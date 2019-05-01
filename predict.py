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
    lang, test_set = prepare_jldata("snli_1.0/snli_1.0_dev.jsonl", lang)

    gls = list()
    pls = list()
    for datapoint in test_set:
        premise = datapoint["premise"]
        hypothesis = datapoint["hypothesis"]
        gold_label = datapoint["gold_label"]
        gls.append(gold_label)
        predict = classifier.get_relation(premise, hypothesis)
        pls.append(predict)
    # correct = 0
    # incorrect = 0
    id2class = {1:"neutral", 2:"contradiction", 3:"entailment"}
    classes = {"neutral":[0,0,0], "contradiction":[0,0,0], "entailment":[0,0,0]}
    # for i, g in enumerate(gls):
    #     if g == pls[i]:
    #         correct += 1
    #     elif g != 0:
    #         incorrect += 1
    #         datapoint = test_set[i]
    #         premise = [lang.words[i] for i in datapoint["premise"]]
    #         hypothesis = [lang.words[i] for i in datapoint["hypothesis"]]
    #         if pls[i] != 0:
    #             print (id2class[g], id2class[pls[i]], premise, hypothesis)
    # print (correct, incorrect)
    for i, g in enumerate(gls):
        if g != 0:
            classes[id2class[g]][0] += 1
            if g == pls[i]:
                classes[id2class[g]][2] += 1
    for i, p in enumerate(pls):
        if p!= 0:
            classes[id2class[p]][1] += 1
    print (classes)
    for c in classes:
        print (c)
        recall = float(classes[c][2]/classes[c][0])
        try:
            precision = float(classes[c][2]/classes[c][1])
        except:
            precision = 0
        try:
            f1 = 2*precision*recall/(precision+recall)
        except:
            f1 = 0
        print ("Recall: %2f Precision: %2f F1: %2f"%(recall, precision, f1))
