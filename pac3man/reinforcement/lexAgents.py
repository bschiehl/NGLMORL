from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class LDQNLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        raise NotImplementedError()

    def getAction(self, state, filter=None, train=False, supervise=False):
        raise NotImplementedError()

    def update(self, state, action, nextState, reward):
        raise NotImplementedError()

    def getPolicy(self, state):
        raise NotImplementedError()


class PacmanLDQNAgent(LDQNLearningAgent):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        raise NotImplementedError()


    def getAction(self, state, filter=None, train=False, supervise=False):
        raise NotImplementedError()


