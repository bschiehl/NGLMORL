from game import *
from learningAgents import ReinforcementAgent
from networks import *

import random,util,math, pickle, time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical

class LDQNLearningAgent(Agent):
    def __init__(self, in_size, action_size, hidden, train_params):
        self.update_steps = train_params.update_steps
        self.epsilon = train_params.epsilon
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.no_cuda = train_params.no_cuda
        self.update_every = train_params.update_every
        self.slack = train_params.slack
        self.reward_size = train_params.reward_size
        self.network = train_params.network

        self.t = 0
        self.discount = 0.99

        self.actions = list(range(action_size))
        self.action_size = action_size
        if self.network == 'DNN':
            self.model = DNN(in_size, (self.reward_size, action_size), hidden)
        elif self.network == 'CNN':
            self.model = CNN(int((in_size / 3) ** 0.5), channels=3, 
                             out_size=(self.reward_size, action_size), convs=hidden, hidden=hidden)
        else:
            print('invalid network specification')
            assert False

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)

        if torch.cuda.is_available() and not self.no_cuda:
            self.model.cuda()


    def getAction(self, state, filter=None, train=False, supervise=False):
        raise NotImplementedError()

    def update(self, state, action, nextState, reward):
        raise NotImplementedError()

    def getPolicy(self, state):
        raise NotImplementedError()


class PacmanLDQNAgent(LDQNLearningAgent, PacmanAgent):
    def __init__(self, **args):
       # LDQNLearningAgent.__init__(self, in_size=, action_size=5, hidden=, **args)
       raiseNotDefined()

        
    def permissible_actions(self, Q):
        permissible_actions = self.actions

        for i in range(self.reward_size):
            Qi = Q[i, :]
            m = max([Qi[a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[a] >= m - r * abs(m)]

    def getAction(self, state, filter=None, train=False, supervise=False):
        raise NotImplementedError()


class LTQLearningAgent(ReinforcementAgent):
    def __init__(self, train_params, initialisation, action_size, double=False, discount=0.99, **args):
        self.slack = train_params.slack
        ReinforcementAgent.__init__(self, **args)

        self.actions=list(Actions._directions.keys())

        if not double:
            self.Q = [{} for _ in range(train_params.reward_size)]
        else:
            self.Qa = [{} for _ in range(train_params.reward_size)]
            self.Qb = [{} for _ in range(train_params.reward_size)]
    
        self.discount = discount

        if isinstance(initialisation, float) or isinstance(initialisation, int):
            self.initialisation = [initialisation for _ in range(train_params.reward_size)]
        else:
            self.initialisation = initialisation
        
        self.double = double
        if double:
            self.update = self.double_Q_update
        elif train_params.lextab_on_policy:
            self.update = self.SARSA_update
        else:
            self.update = self.Q_update

    def getAction(self, state, filter=False, train=False, supervise=False):
        stateStr = str(state)
        self.init_state(stateStr)
        return self.lexicographic_epsilon_greedy(state, filter, train, supervise)

    def init_state(self, state):
        if not self.double:
            if state not in self.Q[0].keys():
                for i, Qi in enumerate(self.Q):
                    Qi[state] = {a: self.initialisation[i] for a in self.actions}
        else:
            if state not in self.Qa[0].keys():
                for i, Qai in enumerate(self.Qa):
                    Qai[state] = {a: self.initialisation[i] for a in self.actions}
                for i, Qbi in enumerate(self.Qb):
                    Qbi[state] = {a: self.initialisation[i] for a in self.actions}

    def lexicographic_epsilon_greedy(self, state, filter=False, train=False, supervise=False):

        permissible_actions = self.getLegalActions(state, filter, train, supervise)
        state = str(state)

        if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            return np.random.choice(permissible_actions)

        if not self.double:
            for Qi in self.Q:
                m = max([Qi[state][a] for a in permissible_actions])
                r = self.slack
                permissible_actions = [a for a in permissible_actions if Qi[state][a] >= m - r * abs(m)]
        else:
            for Qai, Qbi in zip(self.Qa, self.Qb):
                m = max([0.5 * (Qai[state][a] + Qbi[state][a]) for a in permissible_actions])
                r = self.slack
                permissible_actions = [a for a in permissible_actions if
                                       0.5 * (Qai[state][a] + Qbi[state][a] >= m - r * abs(m))]

        return np.random.choice(permissible_actions)
    
    def Q_update(self, state, action, next_state, reward):

        state = str(state)
        next_state = str(next_state)
        self.init_state(state)
        self.init_state(next_state)
        permissible_actions = self.actions

        for i, Qi in enumerate(self.Q):
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]

            alpha = 0.01
            Qi[state][action] = (1 - alpha) * Qi[state][action] + alpha * (reward[i] + self.discount * m)

    def SARSA_update(self, state, action, next_state, reward):

        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions

        for Qi in self.Q:
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]

        ps = []

        for Qi in self.Q:
            for a in self.actions:
                if a in permissible_actions:
                    ps.append((1 - self.epsilon) / len(permissible_actions) + self.epsilon / len(self.actions))
                else:
                    ps.append(self.epsilon / len(self.actions))

        for i, Qi in enumerate(self.Q):
            exp = sum([p * Qi[next_state][a] for p, a in zip(ps, self.actions)])
            target = reward[i] + self.discount * exp
            alpha = 0.01
            Qi[state][action] = (1 - alpha) * Qi[state][action] + alpha * target

    def double_Q_update(self, state, action, next_state, reward):

        done = state.data._win or state.data._lose
        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions
        r = self.slack

        for i, (Qai, Qbi) in enumerate(zip(self.Qa, self.Qb)):

            if np.random.choice([True, False]):

                m = max([Qbi[next_state][a] for a in permissible_actions])
                permissible_actions = [a for a in permissible_actions if Qbi[next_state][a] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qai[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                alpha = 0.01
                Qai[state][action] = (1 - alpha) * Qai[state][action] + alpha * target

            else:

                m = max([Qai[next_state][a] for a in permissible_actions])
                permissible_actions = [i for i in permissible_actions if Qai[next_state][i] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qbi[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                alpha = 0.01
                Qbi[state][action] = (1 - alpha) * Qbi[state][action] + alpha * target

    def save_model(self, root):
        if not self.double:
            with open('{}-model.pt'.format(root), 'wb') as f:
                pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('{}-model-A.pt'.format(root), 'wb') as f:
                pickle.dump(self.Qa, f, pickle.HIGHEST_PROTOCOL)
            with open('{}-model-B.pt'.format(root), 'wb') as f:
                pickle.dump(self.Qb, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, root):
        if not self.double:
            with open('{}-model.pt'.format(root), 'rb') as f:
                self.Q = pickle.load(f)
        else:
            with open('{}-model-A.pt'.format(root), 'rb') as f:
                self.Qa = pickle.load(f)
            with open('{}-model-B.pt'.format(root), 'rb') as f:
                self.Qb = pickle.load(f)


class PacmanLTQAgent(LTQLearningAgent):
    def __init__(self, train_params, initialisation=0, action_size = 5,  double=False, discount=0.99, **args):
        LTQLearningAgent.__init__(self, train_params, initialisation, action_size, double, discount, **args)

    def getAction(self, state, filter=None, train=False, supervise=False):
        action = LTQLearningAgent.getAction(self,state, filter, train, supervise)
        self.doAction(state,action)
        return action


class LA2CLearningAgent(Agent):
    def __init__(self, **args):
        raiseNotDefined()

class PacmanLA2CAgent(LA2CLearningAgent, PacmanAgent):
    def __init__(self, **args):
        raiseNotDefined()