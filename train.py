from nltk.stem.lancaster import LancasterStemmer
import random
import pickle
import nltk
import json
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
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



model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(training), np.array(output), epochs=1000, batch_size=5, verbose=1)
model.save('model/chatbot_model.h5', hist)

print("model created")
