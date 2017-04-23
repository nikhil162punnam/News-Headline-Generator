FN = 'vocabulary-embedding'
seed=42
vocab_size = 40000
embedding_dim = 100
lower = False # dont lower case the text
import pickle
FN0 = 'tokens_sample' # this is the name of the data file which I assume you already have
with open('%s.pkl'%FN0, 'rb') as fp:
    heads, desc, keywords = pickle.load(fp) # keywords are not used in this project


if lower:
    heads = [h.lower() for h in heads]
if lower:
    desc = [h.lower() for h in desc]
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount
vocab, vocabcount = get_vocab(heads+desc)
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word
def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    
    idx2word = dict((idx,word) for word,idx in word2idx.iteritems())

    return word2idx, idx2word
word2idx, idx2word = get_idx(vocab, vocabcount)
fname = 'glove.6B.%dd.txt'%embedding_dim
import os
import sys
from keras.datasets.data_utils import get_file
datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
datadir = os.path.join(datadir_base, 'datasets')
glove_name = os.path.join(datadir, fname)
if not os.path.exists(glove_name):
    path = 'glove.6B.zip'
    path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
    # unzip{datadir}/{path}
