import sentencepiece as spm
import pandas as pd
import numpy as np
import json
import re
import csv
import matplotlib.pyplot as plt 
import os
from os import path
from pandas.plotting import table

load_path = "../../data/tokenization/All.txt"
data = []
with open(load_path, 'r') as f:
    data = f.readlines()

split = np.array_split(data, 5)

sizes = [33, 400, 2000, 10000, 27524]

unk_percent = []

for i in range(len(sizes)):
    copy_data = split.copy()
    test = copy_data.pop(i)
    train = np.concatenate(copy_data).tolist()

    with open("train.txt", 'w+') as f:
        f.write('\n'.join(train))

    spm.SentencePieceTrainer.train('--input=train.txt --model_prefix=m_{1} --vocab_size={1} --model_type=word'.format(i, sizes[i]))

    sp_user = spm.SentencePieceProcessor()
    sp_user.load('m_{0}.model'.format(sizes[i]))

    token_count = 0
    unk_count = 0
    for sent in test:
        result = sp_user.encode_as_ids(sent)
        pieces = sp_user.encode_as_pieces(sent)
        token_count += len(result)
        unk_count += result.count(0)
        unk_count -= pieces.count('<s>')
        unk_count -= pieces.count('</s>')

    unk_percent.append(unk_count / token_count * 100)  
    os.system("rm train.txt")
    os.system("rm m_{0}.model".format(sizes[i]))
    os.system("rm m_{0}.vocab".format(sizes[i]))


df = pd.DataFrame( {'Vocab Size': sizes, "Unk%": unk_percent})
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

table(ax, df, loc='center')  # where df is your data frame

x = plt.gcf()
x.savefig('../../reports/tokenization/unk_percent.png')


