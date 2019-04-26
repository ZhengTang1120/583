import numpy as np
import dynet_config
dynet_config.set(
    mem=10240,
    random_seed=1,
    # autobatch=True
)
import dynet as dy

import pickle

class LSTMClassifier:
    def __init__(self, words, embeds, embeds_dim, hidden_size, projection_size, target_size):
        self.words = words
        self.vocab_size = len(words)
        self.embeds_dim = embeds_dim
        self.hidden_size = hidden_size
        self.embeds = embeds
        self.projection_size = projection_size
        self.target_size = target_size

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        # print(self.trainer.learning_rate)
        if np.any(embeds):
            self.word_embeddings = self.model.lookup_parameters_from_numpy(self.embeds)
        else:
            self.word_embeddings = self.model.add_lookup_parameters((self.vocab_size, self.embeds_dim))
        self.encoder_lstm = dy.BiRNNBuilder(
            2,
            self.embeds_dim,
            self.hidden_size,
            self.model,
            dy.VanillaLSTMBuilder,
        )
        self.Phis = list()
        for i in range(self.projection_size):
            self.Phis.append(self.model.add_parameters((self.hidden_size, self.hidden_size)))
        self.W = self.model.add_parameters((target_size, projection_size))
        self.b = self.model.add_parameters((target_size))

    def save(self, name):
        params = (
            self.words, self.embeds_dim, self.embeds, self.hidden_size, self.projection_size, self.target_size
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = Hyper(*params)
            parser.model.populate(f'{name}.model')
            return parser

    def encode_sentence(self, sentence):
        embeds_sent = [self.word_embeddings[sentence[i]] for i in range(len(sentence))]
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features[-1]

    def train(self, trainning_set):
        loss_chunk = 0
        loss_all = 0
        total_chunk = 0
        total_all = 0
        losses = []
        for datapoint in trainning_set:

            premise = datapoint["premise"]
            hypothesis = datapoint["hypothesis"]
            gold_label = datapoint["gold_label"]

            ep = self.encode_sentence(premise)
            eh = self.encode_sentence(hypothesis)

            Ps = []
            for i in range(self.projection_size):
                Ps.append(self.Phis[i].expr() * ep)
            P = dy.transpose(dy.concatenate_cols(Ps))
            s = P * eh
            y = dy.softmax(self.W.expr() * s + self.b.expr())

            losses.append(-dy.log(dy.pick(y, gold_label)))

            # process losses in chunks
            if len(losses) > 50:
                loss = dy.esum(losses)
                l = loss.scalar_value()
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                losses = []
                loss_chunk += l
                loss_all += l
                total_chunk += 1
                total_all += 1

        # consider any remaining losses
        if len(losses) > 0:
            loss = dy.esum(losses)
            loss.scalar_value()
            loss.backward()
            self.trainer.update()
            dy.renew_cg()
        print(f'loss: {loss_all/total_all:.4f}')

    def get_relation(self, premise, hypothesis):

        ep = self.encode_sentence(premise)
        eh = self.encode_sentence(hypothesis)

        Ps = []
        for i in range(self.projection_size):
            Ps.append(self.Phis[i].expr() * ep)
        P = dy.transpose(dy.concatenate_cols(Ps))
        s = P * eh
        y = dy.softmax(self.W.expr() * s + self.b.expr())
        return np.argmax(y.vec_value())
