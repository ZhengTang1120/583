import random
from utils import *
from model import *
import time

if __name__ == '__main__':
    random.seed(1)
    start = time.time()
    lang, train_set = prepare_jldata("snli_1.0/snli_1.0_train.jsonl")
    lang, dev_set = prepare_jldata("snli_1.0/snli_1.0_dev.jsonl", lang)
    embeds = load_embeddings("glove.6B.200d.txt", lang.words)
    classifier = Classifier(200, embeds, len(lang.words), 100, 48, 4)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.1)
    done = time.time()
    elapsed = done - start
    print("Data Prepared, Time Used: %2f"%(e, elapsed), end=' ')
    print ("training...")
    for e in range(100):
        for i, datapoint in enumerate(train_set):
            classifier.zero_grad()

            premise = torch.LongTensor(datapoint["premise"])
            hypothesis = torch.LongTensor(datapoint["hypothesis"])
            gold_label = torch.LongTensor([datapoint["gold_label"]])
            
            score = classifier(premise, hypothesis)
            loss = loss_function(score, gold_label)

            loss.backward()
            optimizer.step()

        if e % 10 == 0:
            done = time.time()
            elapsed = done - start
            print("Epoch %d: Time Used: %2f"%(e, elapsed), end=' ')
            classifier.eval()
            gls = list()
            pls = list()
            for datapoint in dev_set:
                premise = torch.LongTensor(datapoint["premise"])
                hypothesis = torch.LongTensor(datapoint["hypothesis"])
                gold_label = datapoint["gold_label"]
                gls.append(gold_label)
                predict = np.argmax(classifier(premise, hypothesis).data.numpy())
                pls.append(predict)
            res = np.array([pls[i]==gls[i] for i in range(len(pls))])
            acc = float(np.sum(res)/len(gls))
            print ("Accuracy: %2f"%acc)
            classifier.train()
            start = time.time()
