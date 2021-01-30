from nltk.stem.lancaster import LancasterStemmer
import random
import pickle
import nltk
import json
import tensorflow
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
import warnings
warnings.filterwarnings("ignore")
stemmer = LancasterStemmer()


with open('data/intents.json') as file:
    data = json.load(file)

with open("data/data.pickle", 'rb') as f:
    words, labels, training, output = pickle.load(f)
print("Pickle Loaded")


model = load_model('model/chatbot_model.h5')


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

        pred = model.predict(np.array([bag_of_words(inp, words)]))[0]
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
