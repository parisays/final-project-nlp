import os

genre = [ 'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Romance' ]

for label in genre:
    print(label)
    os.system("cp ../../data/word2vec/{0}.txt utils/datasets/stanfordSentimentTreebank/datasetSentences.txt".format(label))
    os.system("python run.py")
    os.system("mv saved_params_1000.npy ../../models/word2vec/{0}.word2vec.npy".format(label))
    os.system("mv saved_state_1000.pickle ../../models/word2vec/{0}.word2vec.pickle".format(label))

