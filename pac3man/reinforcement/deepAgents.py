from networks import *
from game import *
from learningAgents import ReinforcementAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random, util, time

# base DQN class, extensions should override self.model, self.target_model, and self.optimizer
class DQNLearningAgent(ReinforcementAgent):
    def __init__(self, train_params, action_size, discount, **args):
        ReinforcementAgent.__init__(self, **args)
        self.t = 0
        self.epsilon_start = train_params.epsilon_start
        self.epsilon_decay = train_params.epsilon_decay
        self.tau = train_params.tau
        self.epsilon = train_params.epsilon
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.update_every = train_params.update_every
        self.discount = discount
        self.action_size = action_size
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.episodeLosses = []

        if torch.cuda.is_available() and not train_params.no_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)

    def getAction(self, state, filter=None, train=False, supervise=False):
        permissible_actions = self.getLegalActions(state, filter, train, supervise)

        curr_e = self.epsilon + (self.epsilon_start - self.epsilon) * np.exp(-self.t / self.epsilon_decay) if self.episodesSoFar < self.numTraining else 0
        
        if np.random.choice([True, False], p=[curr_e, 1 - curr_e]):
            random_value = np.random.randint(1, 1 + self.action_size) 
            move = str(util.Action(random_value))
            self.lastAction = move
            if move not in permissible_actions:
                move = Directions.STOP
            return move

        state = util.getStateMatrices(state)
        state = torch.from_numpy(np.stack(state))
        state = state.unsqueeze(0).float()
        state = state.to(self.device)
    
        Qs = self.model(state).detach().cpu().numpy()[0]

        best = np.argwhere(Qs == np.amax(Qs)).flatten()
        move = np.random.choice(best) + 1
        move = str(util.Action(move))
        self.lastAction = move
        if move not in permissible_actions:
            move = Directions.STOP
        return move

    def update(self, state, action, nextState, reward):
        done = nextState.data._win or nextState.data._lose
        self.t += 1
        state = util.getStateMatrices(state)
        nextState = util.getStateMatrices(nextState)
        action = int(util.Action(action)) -1

        if reward > 300:
            reward = 100.
        elif reward > 20:
            reward = 50.
        elif reward > 0:
            reward = 10.
        elif reward < -10:
            reward = -500.
        elif reward < 0:
            reward = -1.
        
    
        self.memory.add(state, action, nextState, reward, int(done))

        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            target_model_state_dict = self.target_model.state_dict()
            policy_model_state_dict = self.model.state_dict()
            for key in policy_model_state_dict:
                target_model_state_dict[key] = self.tau * policy_model_state_dict[key] + (1 - self.tau) * target_model_state_dict[key]
            self.target_model.load_state_dict(target_model_state_dict)

    def learn(self, experiences):
        states, actions, nextStates, rewards, dones = experiences

        self.model.train()
        Qs = self.model(states)
        Qs = Qs.gather(1, actions)

        with torch.no_grad():
            Qs_next = self.target_model(nextStates).detach().max(1)[0]
            Qs_next = Qs_next.unsqueeze(1)

        Q_targets = rewards + self.discount * Qs_next * (1 - dones)

        loss = F.smooth_l1_loss(Qs, Q_targets).to(self.device)
        self.episodeLosses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        self.model.eval()

    def save_model(self, currentit, path='models/'):
        torch.save(self.model.state_dict(), '{}policy-model{}.pt'.format(path, currentit))
        torch.save(self.target_model.state_dict(), '{}target-model{}.pt'.format(path, currentit))

    def load_model(self, path='models/'):
        self.model.load_state_dict(torch.load('{}policy-model-0.pt'.format(path)))
        self.target_model.load_state_dict(torch.load('{}target-model-0.pt'.format(path)))


class PacmanDQNAgent(DQNLearningAgent):
    def __init__(self, train_params, action_size=4, discount=0.99, **args):
        super().__init__(train_params, action_size, discount, **args)
        self.model = PacmanCNN(train_params.width, train_params.height, largeEnv=train_params.largeEnv).to(self.device)
        self.target_model = PacmanCNN(train_params.width, train_params.height, largeEnv=train_params.largeEnv).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_params.learning_rate, amsgrad=True)
        if train_params.trained:
           self.load_model(path=train_params.model_path)
           print("Loaded model")

    def getAction(self, state, filter=None, train=False, supervise=False):
        copy = state.deepCopy()
        action = DQNLearningAgent.getAction(self, state, filter, train, supervise)
        self.lastState = copy
        return action

        
class ACLearningAgent(ReinforcementAgent):
    def __init__(self, train_params, action_size, discount, **args):
        ReinforcementAgent.__init__(self, **args)
        self.t = 0
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.update_every = train_params.update_every
        self.discount = discount
        self.action_size = action_size
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.episodeLosses = []

        if torch.cuda.is_available() and not train_params.no_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.device)

    def getAction(self, state, filter=None, train=False, supervise=False):
        permissible_actions = self.getLegalActions(state, filter, train, supervise)

        state = util.getStateMatrices(state)
        state = torch.from_numpy(np.stack(state))
        state = state.unsqueeze(0).float()
        state = state.to(self.device)
    
        dist = Categorical(self.actor(state))
        move = dist.sample().item() + 1
        move = str(util.Action(move))
        self.lastAction = move
        if move not in permissible_actions:
            move = Directions.STOP
        return move
    
    def update(self, state, action, nextState, reward):
        done = nextState.data._win or nextState.data._lose
        self.t += 1
        state = util.getStateMatrices(state)
        nextState = util.getStateMatrices(nextState)
        action = int(util.Action(action)) -1

        if reward > 300:
            reward = 100.
        elif reward > 20:
            reward = 50.
        elif reward > 0:
            reward = 10.
        elif reward < -10:
            reward = -500.
        elif reward < 0:
            reward = -1.
        
        self.memory.add(state, action, nextState, reward, int(done))

        if self.t % self.batch_size == 0:
            self.update_critic(self.memory.sample(sample_all=True))
            self.update_actor(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

    def update_critic(self, experiences):
        states, _, nextStates, rewards, dones = experiences

        self.critic.train()
        prediction = self.critic(states)

        with torch.no_grad():
            target = rewards + self.discount * self.critic(nextStates).detach() * (1 - dones)

        loss = F.smooth_l1_loss(prediction, target).to(self.device)
        self.episodeLosses.append(loss.item())
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 100)
        self.critic_optimizer.step()

        self.critic.eval()

    def update_actor(self, experiences):
        states, actions, nextStates, rewards, dones = experiences

        self.actor.train()
        with torch.no_grad():
            baseline = self.critic(states)
            outcome = rewards + self.discount * self.critic(nextStates) * (1 - dones)
            advantage = outcome - baseline
        
        dists = self.actor(states)
        log_probs = torch.log(torch.gather(dists, 1, actions))
        loss = -(log_probs * advantage).mean().to(self.device)
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 100)
        self.actor_optimizer.step()

        self.actor.eval()

    def save_model(self, currentit, path='models/'):
        torch.save(self.actor.state_dict(), '{}actor{}.pt'.format(path, currentit))
        torch.save(self.critic.state_dict(), '{}critic{}.pt'.format(path, currentit))

    def load_model(self, path='models/'):
        self.actor.load_state_dict(torch.load('{}actor.pt'.format(path)))
        self.critic.load_state_dict(torch.load('{}critic.pt'.format(path)))

        
class PacmanACAgent(ACLearningAgent):
    def __init__(self, train_params, action_size=4, discount=0.99, **args):
        super().__init__(train_params, action_size, discount, **args)
        self.actor = PacmanActor(train_params.width, train_params.height, largeEnv=train_params.largeEnv).to(self.device)
        self.critic = PacmanCritic(train_params.width, train_params.height, largeEnv=train_params.largeEnv).to(self.device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr= 0.01*train_params.learning_rate, amsgrad=True)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=train_params.learning_rate, amsgrad=True)
        if train_params.trained:
           self.load_model(path=train_params.model_path)
           print("Loaded model")

    def getAction(self, state, filter=None, train=False, supervise=False):
        copy = state.deepCopy()
        action = ACLearningAgent.getAction(self, state, filter, train, supervise)
        self.lastState = copy
        return action