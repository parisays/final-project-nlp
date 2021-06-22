#!/usr/bin/env python

import random
import numpy as np
import pandas as pd
from pandas.plotting import table
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os

from word2vec import *
from sgd import *

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

# Reset the random seed to make sure that everyone gets the same results
# random.seed(314)



genre = [ 'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Romance' ]

all_tokens = []
all_word_vectors = []

for label in genre:
    data_path = "../../data/word2vec/{0}.txt".format(label)
    model_path = "../../models/word2vec/".format(label)

    os.system("cp {0} utils/datasets/stanfordSentimentTreebank/datasetSentences.txt".format(data_path))

    params_file = model_path + "{0}.word2vec.npy".format(label)
    state_file = model_path + "{0}.word2vec.pickle".format(label)
    params = np.load(params_file)
    with open(state_file, "rb") as f:
        state = pickle.load(f)
        # return st, params, state
    
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)
    print("Number of tokens:", nWords)

    # wordVectors = np.concatenate(
    # (params[:nWords,:], params[nWords:,:]),
    # axis=0)

    all_tokens.append(tokens)
    all_word_vectors.append(params)
    # all_word_vectors.append(wordVectors)


elements_in_all = list(set.intersection(*map(set, [list(x.keys()) for x in all_tokens])))
print(len(elements_in_all))
print(elements_in_all[0])

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

for word in elements_in_all:
    tags = []
    comparision = []
    for idx1 in range(len(genre) - 1):
        for idx2 in range(idx1+1, len(genre)):
            # print(genre[idx1], genre[idx2])
            g1 = all_tokens[idx1][word]
            g2 = all_tokens[idx2][word]
            # print(g1)
            similarity = get_cosine_similarity(all_word_vectors[idx1][g1], all_word_vectors[idx2][g2])
            tags.append((genre[idx1], genre[idx2]))
            comparision.append(similarity)
    

    df = pd.DataFrame( { "Tags": tags, "Cosine Similarity": comparision })
    table(ax, df, loc='center')  # where df is your data frame

    x = plt.gcf()
    x.savefig('../../reports/word2vec/{0}.png'.format(word))
    # break
