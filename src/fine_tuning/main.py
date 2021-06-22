import pandas as pd
import numpy as np
import json
import re
import csv
import matplotlib.pyplot as plt 
# import seaborn as sns
from tqdm import tqdm
import ast
import os
from os import path
from pandas.plotting import table
import torch
from torch.utils.data import TensorDataset, random_split
# Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(sentence_df['Description'].values
from datasets import Dataset, DatasetDict
from transformers import pipeline
from MLM import train_MLM

load_path = '../../data/fine_tuning/'
sentence_df = pd.read_pickle(load_path + 'sentence_df.pkl')
sentence_level_df = pd.read_pickle(load_path + 'sentence_level_df.pkl')

sentence_df = sentence_df.reset_index(drop=True)
sentence_level_df = sentence_level_df.reset_index(drop=True)

sentences = [ 'the genre is', 
                'he was from',
                'she used ',
                'she and',
                'in that time', ]

for idx in range(6):
    label = sentence_level_df['Genre'][idx]
    print(label)
    dir = "../../models/fine_tuning/{0}_bert_lm".format(label)

    data = np.concatenate(sentence_level_df['Description'][idx]).tolist()

    train_len = int(len(data)* 0.8)
    # val_len = int(len(data)* 0.1)
    test_len = len(data) - (train_len)
    # train, val, test = torch.utils.data.random_split(data, [train_len, val_len, test_len])
    train, val = torch.utils.data.random_split(data, [train_len, test_len])

    df = pd.DataFrame({"text": [data[idx] for idx in train.indices]})
    train_dataset = Dataset.from_pandas(df)  

    df = pd.DataFrame({"text": [data[idx] for idx in val.indices]})
    val_dataset = Dataset.from_pandas(df)  
                        
    # df = pd.DataFrame({"text": [data[idx] for idx in test.indices]})
    # test_dataset = Dataset.from_pandas(df)  


    dataset = DatasetDict({"train":train_dataset, 
                        "validation":val_dataset, 
                        # "test": test_dataset, 
                       })

    
    model, tokenizer, perplexity = train_MLM(dataset, dir)

    model.save_pretrained(dir)

    file = open(dir+"/perplexity.txt","w+")
    file.write(str(perplexity))
    file.close()

    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    file = open(dir+"/predictions.txt","w+")
    for prefix_text in sentences:
        text = text_generation(prefix_text, max_length=10)
        file.write(str(text)+ os.linesep)

    file.close()


#    break

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
