import os
genre = [ 'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Romance' ]

sentences = [   'human must',
                'four',
                'group', 
                'help',
                'also user',
                'boy',
                'meet',
                'genre',
                ]

for label in genre:
    for sent in sentences:
        os.system("python predict.py --char {0} --input {1}".format(label, sent))