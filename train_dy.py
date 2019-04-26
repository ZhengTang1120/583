import random
from utils import *
from model_dy import *
import time
import pickle

if __name__ == '__main__':
    random.seed(1)
    start = time.time()
    lang, train_set = prepare_jldata("snli_1.0/snli_1.0_train.jsonl")
    lang, dev_set = prepare_jldata("snli_1.0/snli_1.0_dev.jsonl", lang)
    embeds = load_embeddings("glove.6B.200d.txt", lang.words)
    classifier = LSTMClassifier(lang.words, embeds, 200, 100, 48, 4)
    done = time.time()
    elapsed = done - start
    print("Data Prepared, Time Used: %2f"%elapsed)
    print ("training...")
    for i in range(100):
        random.shuffle(train_set)
        classifier.train(train_set)
        if (i % 10) == 0 :
            done = time.time()
            elapsed = done - start
            print("Epoch %d: Time Used: %2f"%(i, elapsed), end=' ')
            gls = list()
            pls = list()
            for datapoint in dev_set:
                premise = datapoint["premise"]
                hypothesis = datapoint["hypothesis"]
                gold_label = datapoint["gold_label"]
                gls.append(gold_label)
                predict = classifier.get_relation(premise, hypothesis)
                pls.append(predict)
            res = np.array([pls[i]==gls[i] for i in range(len(pls))])
            acc = float(np.sum(res)/len(gls))
            print ("Accuracy: %2f"%acc)
            classifier.save("model%d.model"%i)