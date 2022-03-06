import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(10, 3)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN():
    def __init__(
            self,
            n_actions,
            n_features,
            reward_decay=0.9,
            e_greedy=0.9,
            memory_size=2000,
            learning_rate = 0.01,
            batch_size=32,
            replace_target_iter=100,
            e_greedy_increment=None,   
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self._build_net()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate)

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

    def _build_net(self):
        self.evaluate_net = Net()
        self.target_net = Net()

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.evaluate_net(torch.from_numpy(observation).float())
            action = np.argmax(actions_value.detach().numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]      
        temp_s = Variable(torch.FloatTensor(batch_memory[:, :self.n_features]))
        temp_a = Variable(torch.LongTensor(batch_memory[:, self.n_features: self.n_features+1].astype(int)))
        temp_r = Variable(torch.FloatTensor(batch_memory[:, self.n_features+1: self.n_features+2]))
        temp_s_ = Variable(torch.FloatTensor(batch_memory[:, -self.n_features:]))

        actions_value = self.target_net(temp_s_).detach()
        q_value = torch.max(actions_value, dim=1)[0].view(self.batch_size, 1)
        q_target = temp_r + self.gamma * q_value
        q_eval = self.evaluate_net(temp_s)
        q_eval = q_eval.gather(1, temp_a)

        loss = self.criterion(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()