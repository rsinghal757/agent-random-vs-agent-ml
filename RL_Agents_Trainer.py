import numpy as np
import copy
import random
from Agents import AgentML, AgentRandom
from My_Model import baseline_model

def test(test_data, true_labels, model, word_vector_size):

    envML = AgentML()

    for k in range(len(test_data)):  # no. of games(sentences) in one epoch

        sentence_embedding = test_data[k]

        ml_intent_list = []
        ml_reward_list = []

        for l in range(len(sentence_embedding)):  # no. of episodes(words) in one game(sentence)

            ml_reward = -1

            current_state = sentence_embedding[l].reshape(1, word_vector_size)

            ml_dist = envML.act(current_state, model)

            if np.argmax(true_labels[k]) == np.argmax(ml_dist):
                ml_reward = 1

            ml_intent_list.append(ml_dist)
            ml_reward_list.append(ml_reward)

        envML.dist_holder(ml_intent_list, ml_reward_list)

    cum_intent_ml = []
    abs_intent_ml = []

    cum_test_wins = 0
    abs_test_wins = 0

    for k in range(len(envML.intent_dist)):
        cum_intent_ml.append(np.mean(envML.intent_dist[k], axis=0))

    for i in range(len(true_labels)):

        if np.argmax(true_labels[i]) == np.argmax(cum_intent_ml[i]):
            cum_test_wins += 1

    # Here onwards
    abs_dist = copy.deepcopy(envML.intent_dist)
    # abs_dist = envML.intent_dist.copy()

    for k in range(len(abs_dist)):

        for l in range(len(abs_dist[k])):

            temp = envML.intent_dist[k][l][0]
            abs_dist[k][l] = ((temp == temp[np.argmax(temp)]) * 1)

    for k in range(len(envML.intent_dist)):
        abs_intent_ml.append(np.sum(abs_dist[k], axis=0))

    for i in range(len(true_labels)):

        if np.argmax(true_labels[i]) in np.argwhere(abs_intent_ml[i] == np.amax(abs_intent_ml[i])).flatten():
            abs_test_wins += 1

    abs_test_acc = (abs_test_wins/len(test_data)) * 100
    cum_test_acc = (cum_test_wins/len(test_data)) * 100

    return cum_test_acc, abs_test_acc



def train(train_data, labels, epochs, test_data, true_labels, word_vector_size, num_intents, hidden_size):

    model = baseline_model(word_vector_size, num_intents, hidden_size)

    inputs = []
    targets = []

    cum_train_acc_hist = []
    cum_test_acc_hist = []
    abs_train_acc_hist = []
    abs_test_acc_hist = []
    random_wins = []
    
    # Overpopulating start
    max_intent = max(np.unique(labels, axis=0, return_counts=True)[1])
    
    for i in range(num_intents):
            
        tup = np.unique(labels, axis=0, return_counts=True)
        current_intent = tup[0][i]
        current_count = tup[1][i]
        diff_intent = max_intent - current_count
        
        indices = [k for k, x in enumerate(labels) if sum(x == current_intent) == num_intents]
            
        for j in range(diff_intent):
            
            index = random.choice(indices)
            train_data.append(train_data[index])
            labels = labels.tolist()
            labels.append(labels[index])
            labels = np.array(labels)
    # Overpopulating end

    for e in range(epochs):  # epochs

        envRM = AgentRandom()
        envML = AgentML()

        for i in range(len(train_data)):  # no. of games(sentences) in one epoch

            sentence_embedding = train_data[i]

            rm_intent_list = []
            ml_intent_list = []

            rm_reward_list = []
            ml_reward_list = []

            for j in range(len(sentence_embedding)):  # no. of episodes(words) in one game(sentence)

                rm_reward = -1
                ml_reward = -1

                current_state = sentence_embedding[j].reshape(1, word_vector_size)

                rm_dist = envRM.act(num_intents)
                ml_dist = envML.act(current_state, model)

                if np.argmax(labels[i]) == np.argmax(rm_dist):
                    rm_reward = 1

                if np.argmax(labels[i]) == np.argmax(ml_dist):
                    ml_reward = 1

                rm_intent_list.append(rm_dist)
                ml_intent_list.append(ml_dist)

                rm_reward_list.append(rm_reward)
                ml_reward_list.append(ml_reward)

            envRM.dist_holder(rm_intent_list, rm_reward_list)
            envML.dist_holder(ml_intent_list, ml_reward_list)

        cum_reward_rm = []
        cum_reward_ml = []

        for i in range(len(envRM.reward_dist)):
            cum_reward_rm.append(np.mean(envRM.reward_dist[i]))

        for i in range(len(envML.reward_dist)):
            cum_reward_ml.append(np.mean(envML.reward_dist[i]))

        diff = np.array(cum_reward_rm) - np.array(cum_reward_ml)
        rm_wins = (diff > 0)
        random_wins.append(sum(rm_wins))

        for i in range(len(rm_wins)):

            if rm_wins[i] == True:  # Agent Random performed better than Agent ML

                for j in range(len(train_data[i])):

                    if envRM.reward_dist[i][j] == 1:

                        if (train_data[i][j]).tolist() in inputs:
    
                            input_index = inputs.index((train_data[i][j]).tolist())
                            target_index = np.argmax(envRM.intent_dist[i][j])
                            targets[input_index][target_index] += diff[i]
    
                        else:
    
                            inputs.append((train_data[i][j]).tolist())
                            index = np.argmax(envRM.intent_dist[i][j])
                            target = (np.zeros((num_intents,))).tolist()
                            target[index] = diff[i]
                            targets.append(target)

        X_generated = copy.deepcopy(inputs)
        y_generated = copy.deepcopy(targets)

        for i in range(len(y_generated)):
            y_generated[i] = [float(j) / max(y_generated[i]) for j in y_generated[i]]

        model = baseline_model(word_vector_size, num_intents, hidden_size)
        model.fit(np.array(X_generated), np.array(y_generated), epochs=500, batch_size=32)

        cum_train_acc, abs_train_acc = test(train_data, labels, model, word_vector_size)
        cum_test_acc, abs_test_acc = test(test_data, true_labels, model, word_vector_size)

        cum_train_acc_hist.append(cum_train_acc)
        abs_train_acc_hist.append(abs_train_acc)
        cum_test_acc_hist.append(cum_test_acc)
        abs_test_acc_hist.append(abs_test_acc)

        print("Epoch " + str(e) + " Train " + str(cum_train_acc) + " , " + str(abs_train_acc))
        print("Epoch " + str(e) + " Test " + str(cum_test_acc) + " , " + str(abs_test_acc))
            
    
    return model, cum_train_acc_hist, abs_train_acc_hist, cum_test_acc_hist, abs_test_acc_hist, random_wins