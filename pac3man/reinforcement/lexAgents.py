from game import *
from learningAgents import ReinforcementAgent
from networks import *

import random,util,math, pickle, time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical

# LexDQN

class LDQNLearningAgent(ReinforcementAgent):
    def __init__(self, train_params, action_size, discount, **args):
        ReinforcementAgent.__init__(self, **args)
        self.update_steps = train_params.update_steps
        self.epsilon_start = train_params.epsilon_start
        self.epsilon_decay = train_params.epsilon_decay
        self.tau = train_params.tau
        self.epsilon = train_params.epsilon
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.no_cuda = train_params.no_cuda
        self.update_every = train_params.update_every
        self.slack = train_params.slack
        self.slack_start = train_params.slack_start
        self.additive_slack = train_params.additive_slack
        self.reward_size = train_params.reward_size
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.action_size = action_size
        self.t = 0
        self.discount = discount
        self.episodeLosses = []

        if torch.cuda.is_available() and not self.no_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)


    def getAction(self, state, filter=None, train=False, supervise=False):
        permissible_actions = self.getLegalActions(state, filter, train, supervise)

        curr_e = self.epsilon + (self.epsilon_start - self.epsilon) * np.exp(-self.t / self.epsilon_decay) if self.episodesSoFar < self.numTraining else 0
        if self.t % 1000 == 0:
            print("Epsilon: {}".format(curr_e))
        
        if np.random.choice([True, False], p=[curr_e, 1 - curr_e]):
            random_value = np.random.randint(1, 1 + self.action_size) 
            move = str(util.Action(random_value))
            self.lastAction = move
            if move not in permissible_actions:
                self.lastAction = [Directions.STOP, move]
                move = Directions.STOP
            return move

        state = util.getStateMatrices(state)
        state = torch.from_numpy(np.stack(state))
        state = state.unsqueeze(0).float()
        state = state.to(self.device)
    
        Qs = self.model(state).detach().cpu().numpy()[0]

        optimal = self.optimal_actions(Qs)

        move = np.random.choice(optimal) + 1
        move = str(util.Action(move))
        self.lastAction = move
        if move not in permissible_actions:
            self.lastAction = [Directions.STOP, move]
            move = Directions.STOP
        return move
    
    def optimal_actions(self, Q):
        optimal_actions = range(self.action_size)

        for i in range(self.reward_size):
            Qi = Q[i, :]
            m = max([Qi[a] for a in optimal_actions])
            r = self.slack[i]
            if self.additive_slack:
                if self.episodesSoFar < self.numTraining:
                    slack = self.slack_start[i] + (r - self.slack_start[i]) * self.episodesSoFar / self.numTraining
                else:
                    slack = r
                optimal_actions = [a for a in optimal_actions if Qi[a] >= m - slack]
            else:
                optimal_actions = [a for a in optimal_actions if Qi[a] >= m - r * abs(m)]
            
        
        return optimal_actions # from 0
        
    def lexmax(self, Q):
        a = self.optimal_actions(Q)[0]
        return Q[:, a]

    def update(self, state, action, nextState, reward):
        done = nextState.data._win or nextState.data._lose
        self.t += 1
        state = util.getStateMatrices(state)
        nextState = util.getStateMatrices(nextState)
        action = int(util.Action(action)) -1

        if reward[-1] > 300:
            reward[-1] = 100.
        elif reward[-1] > 20:
            reward[-1] = 50.
        elif reward[-1] > 0:
            reward[-1] = 10.
        elif reward[-1] < -10:
            reward[-1] = -500.
        elif reward[-1] < 0:
            reward[-1] = -1.
        
        if reward[0] < 0:
            reward[0] = -500.
    
        self.memory.add(state, action, nextState, reward, int(done))

        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            for _ in range(self.update_steps):
                self.learn(experiences)
            target_model_state_dict = self.target_model.state_dict()
            policy_model_state_dict = self.model.state_dict()
            for key in policy_model_state_dict:
                target_model_state_dict[key] = self.tau * policy_model_state_dict[key] + (1 - self.tau) * target_model_state_dict[key]
            self.target_model.load_state_dict(target_model_state_dict)

    def learn(self, experiences):
        states, actions, nextStates, rewards, dones = experiences
        rewards = rewards.squeeze()

        self.model.train()
        Qs = self.model(states)
        idx = torch.cat((actions, actions), 1).reshape(-1, self.reward_size, 1)
        pred = Qs.gather(2, idx).squeeze()

        with torch.no_grad():
            pred_next = self.target_model(nextStates).detach()
            next_values = torch.stack([self.lexmax(Q) for Q in torch.unbind(pred_next, dim=0)], dim=0)

        Q_targets = rewards + self.discount * next_values * (1 - dones)

        loss = F.smooth_l1_loss(pred, Q_targets).to(self.device)
        self.episodeLosses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        self.model.eval()

    def save_model(self, path='models/'):
        torch.save(self.model.state_dict(), '{}policy-model.pt'.format(path))
        torch.save(self.target_model.state_dict(), '{}target-model.pt'.format(path))

    def load_model(self, path='models/'):
        self.model.load_state_dict(torch.load('{}policy-model.pt'.format(path)))
        self.target_model.load_state_dict(torch.load('{}target-model.pt'.format(path)))




class PacmanLDQNAgent(LDQNLearningAgent):
    def __init__(self, train_params, action_size=4, discount=0.99, **args):
       LDQNLearningAgent.__init__(self, train_params, action_size, discount, **args)
       self.model = PacmanCNN(train_params.width, train_params.height, num_actions= action_size, reward_size=train_params.reward_size, largeEnv=train_params.largeEnv).to(self.device)
       self.target_model = PacmanCNN(train_params.width, train_params.height, num_actions= action_size, reward_size=train_params.reward_size, largeEnv=train_params.largeEnv).to(self.device)
       self.target_model.load_state_dict(self.model.state_dict())
       self.optimizer = optim.AdamW(self.model.parameters(), lr=train_params.learning_rate, amsgrad=True)
       if train_params.trained:
           self.load_model(path=train_params.model_path)
           print("Loaded model")
       if torch.cuda.is_available() and not self.no_cuda:
           self.model.cuda()


    def getAction(self, state, filter=None, train=False, supervise=False):
        copy = state.deepCopy()
        action = LDQNLearningAgent.getAction(self, state, filter, train, supervise)
        self.lastState = copy
        return action
    

# Lexicographic Tabular Q-learning


class LTQLearningAgent(ReinforcementAgent):
    def __init__(self, train_params, initialisation, actions, double=False, discount=0.99, **args):
        self.slack = train_params.slack # 0.001 in experiments
        ReinforcementAgent.__init__(self, **args)

        self.actions=actions
        double = bool(double)
        discount = float(discount)

        if not double:
            self.Q = [{} for _ in range(train_params.reward_size)]
        else:
            self.Qa = [{} for _ in range(train_params.reward_size)]
            self.Qb = [{} for _ in range(train_params.reward_size)]
    
        self.discount = discount

        try:
            initialisation = float(initialisation)
            self.initialisation = [initialisation for _ in range(train_params.reward_size)]
        except TypeError:
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
        i = 0
        if not self.double:
            for Qi in self.Q:
                m = max([Qi[state][a] for a in permissible_actions])
                r = self.slack[i]
                permissible_actions = [a for a in permissible_actions if Qi[state][a] >= m - r * abs(m)]
                i += 1
        else:
            for Qai, Qbi in zip(self.Qa, self.Qb):
                m = max([0.5 * (Qai[state][a] + Qbi[state][a]) for a in permissible_actions])
                r = self.slack[i]
                permissible_actions = [a for a in permissible_actions if
                                       0.5 * (Qai[state][a] + Qbi[state][a] >= m - r * abs(m))]
                i += 1

        return np.random.choice(permissible_actions)
    
    def Q_update(self, state, action, next_state, reward):

        state = str(state)
        next_state = str(next_state)
        self.init_state(state)
        self.init_state(next_state)
        permissible_actions = self.actions
        for i, Qi in enumerate(self.Q):
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack[i]
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]

            Qi[state][action] = (1 - self.alpha) * Qi[state][action] + self.alpha * (reward[i] + self.discount * m)

    def SARSA_update(self, state, action, next_state, reward):

        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions
        i = 0
        for Qi in self.Q:
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack[i]
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]
            i += 1

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
           
            Qi[state][action] = (1 - self.alpha) * Qi[state][action] + self.alpha * target

    def double_Q_update(self, state, action, next_state, reward):

        done = next_state.data._win or next_state.data._lose
        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions
        

        for i, (Qai, Qbi) in enumerate(zip(self.Qa, self.Qb)):
            r = self.slack[i]
            if np.random.choice([True, False]):

                m = max([Qbi[next_state][a] for a in permissible_actions])
                permissible_actions = [a for a in permissible_actions if Qbi[next_state][a] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qai[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                Qai[state][action] = (1 - self.alpha) * Qai[state][action] + self.alpha * target

            else:

                m = max([Qai[next_state][a] for a in permissible_actions])
                permissible_actions = [i for i in permissible_actions if Qai[next_state][i] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qbi[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                Qbi[state][action] = (1 - self.alpha) * Qbi[state][action] + self.alpha * target

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
    def __init__(self, train_params, initialisation=0, double=False, discount=0.99, **args):
        LTQLearningAgent.__init__(self, train_params, initialisation, list(Actions._directions.keys()), double, discount, **args)

    def getAction(self, state, filter=None, train=False, supervise=False):
        action = LTQLearningAgent.getAction(self,state, filter, train, supervise)
        self.doAction(state,action)
        return action

# Lexicographic policy-based reinforcement learning

class LA2CLearningAgent(Agent):
    def __init__(self, **args):
        raiseNotDefined()

class PacmanLA2CAgent(LA2CLearningAgent, PacmanAgent):
    def __init__(self, **args):
        raiseNotDefined()