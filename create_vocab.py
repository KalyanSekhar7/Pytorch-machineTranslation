from collections import Counter
import torch
from nltk import word_tokenize
from tqdm import tqdm
from torch.utils.data import Dataset

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
    corpus.append(torch.full((padding_length,), word2idx["<PAD>"]))
    padded_corpus = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in corpus], batch_first=True,
                                                    padding_value=word2idx["<PAD>"])
    padded_corpus = padded_corpus[1:]
    # changing the padded corupus shape from (batch_size,target_len) ->(target_len,batch_size)
    padded_corpus = padded_corpus.transpose(0, 1)
    print(padded_corpus.shape)
    # print("final padded sequence length is ", padded_corpus.shape)
    # print(word2idx, padded_corpus)
    return word2idx, padded_corpus


class CreateDataset(Dataset):

    def __init__(self):
        french_corpus, french_words, english_corpus, english_words = MakeCorpus("fra-eng/fra.txt").create_corpus()

        self.french_dict, self.french_vec_corpus = create_vocabulary(french_words, french_corpus)
        self.english_dict, self.english_vec_corpus = create_vocabulary(english_words, english_corpus)

    def __len__(self):
        return len(self.french_vec_corpus)

    def __getitem__(self, index):
        return self.english_vec_corpus[index], self.french_vec_corpus[index]
