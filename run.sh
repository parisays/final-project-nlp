#!/bin/sh

#word2vec
cd src/word2vec
python main.py
python compare.py

#tokenization
cd ../tokenization
python main.py
python best_model.py

#parsing
cd ../parsing
python run.py

#language_model
cd ../language_model
python main.py

#fine_tune
cd ../fine_tune
python main.py
python generate.py