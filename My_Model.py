from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def baseline_model(word_vector_size, num_intents, hidden_size):

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(word_vector_size,), activation='relu'))
    model.add(Dense(num_intents, activation="softmax"))
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer=adam, loss='mean_squared_error')

    return model