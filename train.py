from nltk.stem.lancaster import LancasterStemmer
import random
import pickle
import nltk
import json
import tensorflow
import tflearn
import numpy as np
import warnings
warnings.filterwarnings("ignore")
stemmer = LancasterStemmer()

with open('data/intents.json') as file:
    data = json.load(file)

try:
    with open("data/data.pickle", 'rb') as f:
        words, labels, training, output = pickle.load(f)
    print("Pickle Loaded")
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs_x):
        bag = []
        _wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in _wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1

        training.append(bag)
        output.append(output_row)

        with open("data/data.pickle", 'wb') as f:
            pickle.dump((words, labels, training, output), f)
        print("Pickle Saved")


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model/model.tflearn")
print("Model Saved")
