import json
import string
import random 

import nltk 
import numpy as np 
import pickle

from nltk.stem import WordNetLemmatizer


import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout

resources = ["punkt", "punkt_tab", "wordnet"]
for r in resources:
    try:
        nltk.data.find(r)
    except LookupError:
        nltk.download(r)

data_file = open('./intents.json').read()

data = json.loads(data_file)


# Build vocabulary (`words`) and training labels (`classes`) from intents.json

words=[]
classes=[]
data_x=[]
data_y=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Split each sentence into tokens for vocabulary creation
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        # Keep raw pattern and its tag so we can vectorize later
        data_x.append(pattern)
        data_y.append(intent["tag"])

    # Ensure each class tag appears only once
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
    

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(word) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))



# Convert text + labels to numeric vectors for model training


training = []
# Template for one-hot encoded label vector
out_empty = [0]*len(classes)

for idx, doc in enumerate(data_x):
    bow=[]
    # Normalize text to improve matching against vocabulary
    tokens = nltk.word_tokenize(doc.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    for word in words:
        bow.append(1 if word in tokens else 0)


    # One-hot encode the current intent tag
    out_row=list(out_empty)
    out_row[classes.index(data_y[idx])]=1
    training.append([bow, out_row])


# Shuffle so training order does not follow intents.json grouping
random.shuffle(training)
training = np.array(training, dtype=object)

# Split feature vectors (X) and labels (y)
train_x=np.array(list(training[:, 0]))
train_y=np.array(list(training[:, 1]))

model = Sequential()

# Input layer expects one value per vocabulary word
model.add(Dense(128, input_shape = (len(train_x[0]), ), activation='relu'))

# Dropout helps reduce overfitting by randomly disabling neurons during training
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

# Output layer size equals number of intent classes
model.add(Dense(len(train_y[0]), activation='softmax'))

adam = tf.keras.optimizers.Adam(learning_rate = 0.01)

model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

print('Summary', model.summary())

# Train model on vectorized patterns and one-hot labels
model.fit(x=train_x, y=train_y, epochs=150, verbose=1)

model.save('chatbot_model.keras')

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)


