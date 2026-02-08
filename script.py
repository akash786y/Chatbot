import pickle
import nltk
import numpy as np
import tensorflow as tf
import random
from nltk.stem import WordNetLemmatizer

import json

data_file = open('./intents.json').read()
data = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

model = tf.keras.models.load_model('chatbot_model.keras')

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)


def cleanText(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def bagOfWords(text, vocab):
    tokens = cleanText(text)
    bow = [1 if w in tokens else 0 for w in words]
    
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bagOfWords(text, vocab)
    print("BOW sum:", np.sum(bow))
    result = model.predict(np.array([bow]))[0]
    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res>thresh]
    y_pred.sort(key=lambda x:x[1], reverse=True)

    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list)==0:
        result = "Sorry! I don't understand"
    else:
        tag = intents_list[0]
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag']==tag:
                result = random.choice(i['responses'])
                break
    return result

while True:
    message = input('')
    if message=='0':
        break
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)