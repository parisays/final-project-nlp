import os
genre = [ 'Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Romance' ]

for label in genre:
    print(label)
    os.system("python train.py --char {0}".format(label))