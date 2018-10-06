import copy
import spacy
import matplotlib.pyplot as plt
from RL_Agents_Trainer import train

nlp = spacy.load('en')

input_file = open('games.txt', 'r+')

with input_file as f:
    lines = [line.rstrip("\n").split("|") for line in f.readlines()]
    labels = []
    input_data = []
    for line in lines:
        labels.append(line[0])
        input_data.append(line[1])
    
    
for i in range(len(input_data)):
    doc = nlp(input_data[i])
    input_data[i] = []

    for token in doc:
        input_data[i].append(token.text)

"""
print("Getting word2vec vectors\n")
from gensim.models import Word2Vec

trained_model = Word2Vec(input_data, min_count=1, size=50)
print(trained_model)


# ==========================================================================

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
filename = 'glove.6B.50d.txt.word2vec'
pre_trained_model = KeyedVectors.load_word2vec_format(filename, binary=False)

# ==========================================================================

for i in range(len(input_data)):

    for j in range(len(input_data[i])):

        # input_data[i][j] = trained_model[input_data[i][j]]

        if input_data[i][j] in list(pre_trained_model.wv.vocab):
            input_data[i][j] = pre_trained_model[input_data[i][j]]
        else:
            input_data[i][j] = trained_model[input_data[i][j]]
"""


from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(array(labels))

#One Hot encoding start
corpus = [item for sublist in input_data for item in sublist]
vocab = np.array(sorted(list(set(corpus))))

vocab_encoder = np.array(range(1, len(vocab) + 1))

onehot_encoder = OneHotEncoder(sparse=False)
vocab_encoder = vocab_encoder.reshape(len(vocab_encoder), 1)
onehot_encoded = onehot_encoder.fit_transform(vocab_encoder)

word_to_hot = {}
for i in range(len(vocab)):
    word_to_hot[vocab[i]] = onehot_encoded[i]

one_hot = copy.deepcopy(input_data)

for i in range(len(one_hot)):

    for j in range(len(one_hot[i])):

        one_hot[i][j] = word_to_hot[one_hot[i][j]]
# One hot encoding ends
        

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(one_hot, integer_encoded,
                                                    test_size=0.2, random_state=42, stratify=integer_encoded)


onehot_encoder = OneHotEncoder(sparse=False)
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)
onehot_encoder.fit(integer_encoded.reshape(-1, 1))
y_train = onehot_encoder.transform(y_train)
y_test = onehot_encoder.transform(y_test)

# parameters
epsilon = .33  # exploration
num_intents = len(set(labels))  # [cancellation, timing, address, etc....]
max_memory = 100  # Maximum number of experiences we are storing
hidden_size = len(set(labels))  # Size of the hidden layers
word_vector_size = len(vocab)  # Size of the word embedding/encoding

import numpy as np

print("Training the model\n")
epochs = 10
model, train_cum, train_abs, test_cum, test_abs, random_agent_wins = train(X_train, y_train, epochs, X_test, y_test,
                                                                           word_vector_size, num_intents, hidden_size)

#import pickle
#
#pickle_out = open("dict.pickle","wb")
#pickle.dump(word_to_hot, pickle_out)
#pickle_out.close()

"""
plt.plot(train_cum)
plt.plot(test_cum)
plt.show()
print(max(train_cum))
print(max(test_cum))
plt.plot(random_agent_wins)
"""