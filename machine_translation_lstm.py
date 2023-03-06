
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import random
from collections import Counter
from nltk.tokenize import  word_tokenize
from tqdm import tqdm
from configparser import ConfigParser


from create_vocab import MakeCorpus,create_vocabulary,CreateDataset
from translation_model import Encoder,Decoder,Seq2Seq

config = ConfigParser()
config.read("lstm_config.ini")
training_data = config["TRAINING"]
model_data = config["MODEL"]
encoder_data = config["ENCODER"]
decoder_data = config["DECODER"]
padding_length = 65

# English to French !!
# tokenize
# Creating the corpus

french_corpus, french_words, english_corpus, english_words = MakeCorpus("fra-eng/fra.txt").create_corpus()
french_dict, french_vec_corpus = create_vocabulary(french_words, french_corpus)
english_dict, english_vec_corpus = create_vocabulary(english_words, english_corpus)

# Defining the parametets


num_epochs = training_data.getint("num_epochs")
learning_rate = training_data.getfloat("learning_rate")
batch_size = training_data.getint("batch_size")

load_model = model_data.getboolean("load_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size_encoder = len(english_dict)
vocab_size_decoder = len(french_dict)
output_size = len(french_dict)

encoder_embedding_size = encoder_data.getint("encoder_embedding_size")
decoder_embedding_size = decoder_data.getint("decoder_embedding_size")
hidden_size = encoder_data.getint("hidden_size")
num_layers = encoder_data.getint("num_layers")
encoder_dropout = encoder_data.getfloat("encoder_dropout")
decoder_dropout = decoder_data.getfloat("decoder_dropout")


# writer = SummaryWriter(f"runs/loss_plot"





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
            # some Exception is coming in concat making it a 0 in a vec dimension
        # to make sure gradients do not explode
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    target_sentence = "I am going outside to play with my friends"
