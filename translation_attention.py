"""
1. Adding Bi directional in the Encoder
    so there cell size = cell size*2
            hidden size = hidden size*2

            a. you take the mean of the bidirectional (.mean() along  the axis
            b.  Make a Fully connected layers of shape ( cell size*2 ->cell size) and let the fc layer decide what to take

"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k  # german to english translator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IWSLT2017
from torch.utils.data import Dataset, DataLoader
# import spacy

from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer

# English to French !!
# tokenize
padding_length = 65


class MakeCorpus:

    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        self.english_words = set()
        self.french_words = set()
        self.eng_corpus = []
        self.french_corpus = []
        for line in tqdm(lines[-50:-1]):
            english_sentence, french_sentence, _ = line.split("\t")
            self.eng_corpus.append(word_tokenize(english_sentence))
            self.french_corpus.append(word_tokenize(french_sentence))
            for word in word_tokenize(english_sentence):
                self.english_words.add(word)
            for word in word_tokenize(french_sentence):
                self.french_words.add(word)

    def create_corpus(self):
        return self.french_corpus, self.french_words, self.eng_corpus, self.english_words


def create_vocabulary(words, corpus, padding_length=65):
    words.update(["<SOS>", "<EOS>", "<UNK>", "<PAD>"])
    vocab = Counter(words)
    vocab = sorted(vocab, key=vocab.get, reverse=True)

    # map words to unique indices
    word2idx = {word: ind for ind, word in enumerate(vocab)}

    for index, sentence in enumerate(corpus):
        corpus[index] = [word2idx["<SOS>"]] + [word2idx[word] for word in sentence] + [word2idx["<EOS>"]]
    # remember to pad the sequence !!!
    # padding the first index to padding length
    # make the last element of the list a pytorch of constant_length
    corpus.append(torch.full((65,), word2idx["<PAD>"]))
    padded_corpus = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in corpus], batch_first=True,
                                                    padding_value=word2idx["<PAD>"])
    padded_corpus = padded_corpus[1:]
    # changing the padded corupus shape from (batch_size,target_len) ->(target_len,batch_size)
    padded_corpus = padded_corpus.transpose(0, 1)
    print(padded_corpus.shape)
    # print("final padded sequence length is ", padded_corpus.shape)
    # print(word2idx, padded_corpus)
    return word2idx, padded_corpus


french_corpus, french_words, english_corpus, english_words = MakeCorpus("fra-eng/fra.txt").create_corpus()

french_dict, french_vec_corpus = create_vocabulary(french_words, french_corpus)
english_dict, english_vec_corpus = create_vocabulary(english_words, english_corpus)


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout_precent):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.dropout = nn.Dropout(dropout_precent)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                            bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape : (seq_length,batch_size)
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        # embedding shape :(seq_length,batch_size,embedding_size)
        encoder_states, (hidden, cell) = self.lstm(embedding)
        # we only care about the context vector so to say for this basic lstm
        """
        Hidden and cell now will have one extra dimension because of bidirectional
        hidden shape : (2,N,hidden_size)
        encoder states -> states of all the timesteps of lstm *2 ( for bidirectional)
        """

        hidden = self.fc_hidden(torch.concat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.cell_hidden(torch.concat((cell[0:1], cell[1:2]), dim=2))
        return encoder_states, hidden, cell


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, num_layers, dropout_precent):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_percent = dropout_precent

        # self.dropout = nn.Dropout(self.dropout_percent)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)
        """
        sending hidden_size*2 ( from the encoder states and the embeddings)
        """
        # hidden size of encoder and decoder should be the same

        self.energy = nn.Linear(hidden_size * 3, 1)  # hidden size from the encoder + hidden from the previous timestep
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        # shape of x is (batch_size) , but we want (1,N) to work word by word ( 1 here represents that its a single word

        x = x.unsqueeze(0)
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        # embedding shape  = (1,N,embedding_size)
        sequence_length = encoder_states.shape[0]
        hidden_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((hidden_reshaped, encoder_states), dim=2)))

        attention = self.softmax(energy)  # (seq_len,N,1)
        attention = attention.permute(1, 2, 0)  # (N,1,seq_len)

        encoder_states = encoder_states.permute(1, 0, 2)
        # shape of output : (1,N,hidden_size)

        # (N,1,hidden_size*2) using torch bmm --> (1,N,hidden_size*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)
        rnn_input = torch.cat((context_vector, embedding), dim=2)

        outputs, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

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
        encoder_states, hidden, cell = self.encoder(source)

        # grab the start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            # output shappe (batch_size,english vocab size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        print("final_output shape:", outputs.shape)
        return outputs


# Defining the parametets

num_epochs = 10
learning_rate = 0.001
batch_size = 16

load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size_encoder = len(english_dict)
vocab_size_decoder = len(french_dict)
output_size = len(french_dict)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
encoder_dropout = 0.5
decoder_dropout = 0.5


# writer = SummaryWriter(f"runs/loss_plot"

class CreateDataset(Dataset):

    def __init__(self):
        french_corpus, french_words, english_corpus, english_words = MakeCorpus("fra-eng/fra.txt").create_corpus()

        self.french_dict, self.french_vec_corpus = create_vocabulary(french_words, french_corpus)
        self.english_dict, self.english_vec_corpus = create_vocabulary(english_words, english_corpus)

    def __len__(self):
        return len(french_vec_corpus)

    def __getitem__(self, index):
        return self.english_vec_corpus[index], self.french_vec_corpus[index]


final_dataset = CreateDataset()

train_loader = DataLoader(dataset=final_dataset, batch_size=batch_size, shuffle=True)

encoder_net = Encoder(vocab_size=vocab_size_encoder, embedding_size=encoder_embedding_size,
                      hidden_size=hidden_size, num_layers=num_layers, dropout_precent=encoder_dropout).to(device)

decoder_net = Decoder(vocab_size=vocab_size_decoder, embedding_size=decoder_embedding_size,
                      hidden_size=hidden_size, output_size=vocab_size_decoder, num_layers=num_layers,
                      dropout_precent=decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, output_size).to(device)
# print(french_dict)
criterion = nn.CrossEntropyLoss(ignore_index=english_dict["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# do the load translation_model.py thing

for epoch in tqdm(range(num_epochs)):
    # print(f"Epoch : {epoch}")

    for batch_idx, (source, target) in enumerate(train_loader):
        source = source.to(device)
        target = target.to(device)

        output = model(source, target)
        # output shae is (target_len,batch_size,output_dim)
        # print("translation_model.py output shape :",output.shape)
        output = output[1:].reshape(-1, output.shape[2])  # like concatenating
        print("translation_model.py output shape2 :", output.shape)
        # print("target",target)
        # print("output",output)
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        try:
            loss.backward()
        except Exception as e:
            print("Error: epoch:", epoch, "batch index:", batch_idx)
        # to make sure gradients do not explode
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    target_sentence = "I am going outside to play with my friends"
