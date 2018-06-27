import numpy as np
from numpy import linalg as LA
import cPickle as pickle
import random
from collections import Counter

class Vocabulary(object):
    def __init__(self, vocab_file, emb_file='', dim_emb=0):
        """
        load vocab and either use random embeddings , or load them from file
        :param vocab_file:
        :param emb_file:
        :param dim_emb:
        """
        with open(vocab_file, 'rb') as f:
            self.size, self.word2id, self.id2word = pickle.load(f)
        self.dim_emb = dim_emb
        self.embedding = np.random.random_sample(
            (self.size, self.dim_emb)) - 0.5

        if emb_file:
            print 'Loading word vectors from', emb_file
            with open(emb_file) as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]])
                    if word in self.word2id:
                        self.embedding[self.word2id[word]] = vec

        for i in range(self.size):
            self.embedding[i] /= LA.norm(self.embedding[i])

def build_vocab(data, path, min_occur=5):
    """
    build a tuple (vocab_size, word2id, id2word) and pickle it to path
    vocab_size is an int word2id is a dictionary. id2word is a list.
    :param data:
    :param path:
    :param min_occur:
    :return: None
    """
    word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
    id2word = ['<pad>', '<go>', '<eos>', '<unk>']

    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)
    with open(path, 'wb') as f:
        pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)
