import json
from nltk.tokenize import word_tokenize
import re
import numpy as np
import math

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.words = ["UNKNOWN"]
        self.n_words = 1

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.words.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1 

def sanitizeWord(w):
    w = w.lower()
    if is_number(w):
        return "xnumx"
    w = re.sub("[^a-z_]+","",w)
    if w:
        return w

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def load_embeddings(file, words):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            x = np.array([float(i) for i in line_split[1:]])
            vector = (x /np.linalg.norm(x))
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
    base = math.sqrt(6/embedding_size)
    emb_matrix = np.random.uniform(-base,base,(len(words), embedding_size))
    for i in range(1, len(words)):
        word = words[i]
        if word in emb_dict:
            emb_matrix[i] = emb_dict[word]
    return emb_matrix

def prepare_jldata(file, lang=None):

    langisnew = False
    if lang is None:
        lang = Lang("SNLI")
        langisnew = True
    training_set = list()
    label_mapping = {"-":0, "neutral":1, "contradiction":2, "entailment":3}

    for line in open(file):
        jl = json.loads(line)

        premise = [sanitizeWord(w) for w in word_tokenize(jl["sentence1"])]
        hypothesis = [sanitizeWord(w) for w in word_tokenize(jl["sentence2"])]
        if langisnew:
            lang.addSentence(premise)
            lang.addSentence(hypothesis)
        gold_label = jl["gold_label"]

        training_set.append(
            {
                "premise": [lang.word2index[w] if w in lang.word2index else 0 for w in premise],
                "hypothesis": [lang.word2index[w] if w in lang.word2index else 0 for w in hypothesis],
                "gold_label": label_mapping[gold_label]
            }
        )

    return lang, training_set

