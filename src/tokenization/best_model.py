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

data_path = "../../data/tokenization/All.txt"
# os.system("cp {0} train.txt")
spm.SentencePieceTrainer.train('--input={0} --model_prefix=tokenization --vocab_size=27524 --model_type=word'.format(data_path))

# sp_user = spm.SentencePieceProcessor()
# sp_user.load('tokenization.model')

os.system("mv tokenization.model ../../models/tokenization/")
os.system("mv tokenization.vocab ../../models/tokenization/")



