import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMEncoder(nn.Module):

    def __init__(self, embedding_dim, embeddings, vocab_size, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        if embeddings:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return lstm_out

class Classifier(nn.Module):

    def __init__(self, embedding_dim, embeddings, vocab_size, hidden_dim, projection_size, target_size):
        super(Classifier, self).__init__()
        self.encoder = LSTMEncoder(embedding_dim, embeddings, vocab_size, hidden_dim)
        # Initialize projection matrices using scheme from Glorot &
        # Bengio (2008).
        var = 2 / (hidden_dim + hidden_dim)
        mat_data = torch.FloatTensor(projection_size, hidden_dim, hidden_dim)
        mat_data.normal_(0, var)
        mat_data += torch.cat([torch.eye(hidden_dim).unsqueeze(0) for _ in range(projection_size)])
        self.pmats = nn.Parameter(mat_data)

        self.fc = nn.Linear(projection_size, target_size)
        

    def forward(self, premise, hypothesis):
        p_vec = self.encoder(premise)[-1]
        h_vec = self.encoder(hypothesis)[-1]


        p = torch.matmul(self.pmats, p_vec.transpose(0,1))
        p = p[:,:,-1]
        h_vec = h_vec.transpose(0,1)
        features = torch.mm(p, h_vec).transpose(1,0)
        output = F.softmax(self.fc(features[-1,:]), dim=0).view(1, -1)

        return output