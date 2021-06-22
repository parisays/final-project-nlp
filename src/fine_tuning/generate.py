from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
import pandas as pd
import os

load_path = './data/'
sentence_df = pd.read_pickle(load_path + 'sentence_df.pkl')
sentence_level_df = pd.read_pickle(load_path + 'sentence_level_df.pkl')

sentence_df = sentence_df.reset_index(drop=True)
sentence_level_df = sentence_level_df.reset_index(drop=True)

sentences = [   'it was clear',
                'she and her friend',
                'in that city', 
                'many years ago',
                'the world ',
                'people from',
                'new places ',]

for idx in range(6):
    label = sentence_level_df['Genre'][idx]
    print(label)
    dir = "{0}_bert_lm".format(label)
    model_checkpoint = "distilgpt2"
    model = AutoModelForCausalLM.from_pretrained(dir)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    file = open(dir+"/predictions.txt","a+")
    for prefix_text in sentences:
        text = text_generation(prefix_text, max_length=10)
        file.write(str(text)+ os.linesep)

    file.close()