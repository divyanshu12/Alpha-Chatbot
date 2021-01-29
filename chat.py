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

with open("data/data.pickle", 'rb') as f:
    words, labels, training, output = pickle.load(f)
print("Pickle Loaded")


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load("model/model.tflearn")
print("Model Loaded")


def bag_of_words(s, words):
    s_words = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(s)]
    bag = [0 for _ in range(len(words))]

    for s_w in s_words:
        for i, w in enumerate(words):
            if w == s_w:
                bag[i] = 1
    return np.array(bag)


def Chat():
    print("Start Chatting with Bot! (quit to stop)")
    while True:
        inp = input("You : ")
        if inp.lower() == 'quit':
            break

        pred = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(pred)
        tag = labels[result_index]

        if pred[result_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    response = tg['responses']
            print(random.choice(response))
        else:
            print("I didn't get that, try again.")


Chat()
