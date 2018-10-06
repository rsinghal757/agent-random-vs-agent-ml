import numpy as np

class AgentRandom(object):

    def __init__(self):
        self.intent_dist = []
        self.reward_dist = []

    def act(self, no_intents):
        dist = np.random.randn(no_intents)
        return dist

    def dist_holder(self, sent_dist, reward_list):
        self.intent_dist.append(sent_dist)
        self.reward_dist.append(reward_list)


class AgentML(object):

    def __init__(self):
        self.intent_dist = []
        self.reward_dist = []

    def act(self, game_state, my_model):
        dist = my_model.predict(game_state)
        return dist

    def dist_holder(self, sent_dist, reward_list):
        self.intent_dist.append(sent_dist)
        self.reward_dist.append(reward_list)