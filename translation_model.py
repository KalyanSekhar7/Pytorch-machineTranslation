import random

import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_precent):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_precent)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_precent)

    def forward(self, x):
        # x shape : (seq_length,batch_size)
        embedding = self.dropout(self.embedding(x))
        # embedding shape :(seq_length,batch_size,embedding_size)
        outputs, (hidden, cell) = self.lstm(embedding)
        # we only care about the context vector so to say for this basic lstm
        return hidden, cell


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout_precent):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_percent = dropout_precent

        self.dropout = nn.Dropout(self.dropout_percent)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_precent)
        # hidden size of encoder and decoder should be the same
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x is (batch_size) , but we want (1,N) to work word by word ( 1 here represents that its a single word

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape  = (1,N,embedding_size)

        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        # shape of output : (1,N,hidden_size)

        prediction_logits = self.fc(outputs)
        # shape of predictions is going to be (1,N, length_of_vocabulary)
        # now remove the 1
        prediction = prediction_logits.squeeze(0)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, decoder_vocab_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.decoder_vocab_size = decoder_vocab_size

    def forward(self, source, target, teacher_force_ratio=0.5):
        # change the target shape t from (batch_size,target_len) to (target_len,batch_size)

        batch_size = source.shape[1]
        # source = (target_len,batchsize)

        target_len = target.shape[0]
        target_vocab_size = self.decoder_vocab_size
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        # grab the start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            # output shappe (batch_size,english vocab size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        print("final_output shape:", outputs.shape)
        return outputs
