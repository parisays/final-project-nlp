import pandas as pd
import numpy as np
import json
import re
import csv
import matplotlib.pyplot as plt 
from tqdm import tqdm
import ast
import os
from pandas.plotting import table
import string
import torch
import sys
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import EarlyStoppingCallback
import math


global block_size
block_size = 128


def train_MLM(dataset, dir):
    def tokenize_function(examples):
        return tokenizer(examples["text"])


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    model_checkpoint = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
            )

    # model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
    dir,
    do_train= True,
    do_eval = True,
    # do_predict = True,
    num_train_epochs = 6,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    #data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    p = math.exp(eval_results['eval_loss'])
    print(f"Perplexity: {p:.2f}")

    return model, tokenizer, p


