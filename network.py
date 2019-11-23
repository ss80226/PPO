import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions
import numpy as np

class Network(nn.Module):
    '''
    input: observation space with dimension (24,)
    output: mean & standard_deviation for each action dimension (4, )  and one state_value
    '''
    def __init__(self, args):
        super(Network, self).__init__()
        self.action_dim = args['action_dim']
        self.state_dim = args['state_dim']
        self.isTrain = args['is_train']
        self.fc1 = nn.Linear(args['state_dim'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.outLayer = nn.Linear(128, args['action_dim']*2) # action_dim * (mean, standard_deviation)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.outLayer(x)
        # print(x.shape)
        # print(x)
        if self.isTrain == False:
            mu_vector = x[0:self.action_dim]
            sigma_vector = x[self.action_dim:self.action_dim*2]
        else:
            mu_vector = x[:, 0:self.action_dim]
            sigma_vector = x[:, self.action_dim:self.action_dim*2]
        mu_vector = torch.tanh(mu_vector)
        # print(mu_vector)
        sigma_vector = F.softplus(sigma_vector)
        
        return mu_vector, sigma_vector
    def act(self, mu_vector, sigma_vector):
        '''
        input: (1, action_dim) mean vecotor
               (1, action_dim) standard_deviation vector
        output: (1, action_dim) action vector
        '''
        action_vector = torch.distributions.normal.Normal(mu_vector, sigma_vector).sample()
        action_vector = torch.clamp(action_vector, -1., 1.) # clipping value into the a ~ (action_space.low, action_space.high)
        # print()
        return action_vector
    def logp(self, state, action):
        mu_vector, sigma_vector = self.forward(state)
        dist = torch.distributions.normal.Normal(mu_vector, sigma_vector)
        logp_vector = dist.log_prob(action)
        # print(logp_vector)
        logp_joint = logp_vector.sum(dim=1, keepdim=True)
        # print(logp_joint)
        # print(logp_joint)
        return logp_joint
# class LSTM(nn.Module):
    # def __init__(self, args):
    #     super(LSTM, self).__init__()
    #     self.action_dim = args['action_dim']
    #     self.hidden_layer_dim = 64
    #     self.hidden_layer_num = 3
    #     self.isTrain = args['is_train']
    #     self.state_dim = args['state_dim']
    #     self.isTrain = args['is_train']
    #     self.hidden_history = None
    #     # self.rnn = nn.LSTM(    
    #     #     input_size = self.state_dim,     
    #     #     hidden_size = self.hidden_layer_dim,     
    #     #     num_layers = self.hidden_layer_num
    #     #     # batch_first=True,   #(batch, time_step, input_size)
    #     # )
    #     self.rnn = nn.GRUCell(input_size = self.state_dim, hidden_size = self.hidden_layer_dim)
    #     self.out = nn.Linear(self.hidden_layer_dim, self.action_dim*2)
    # def reset(self):
    #     self.hidden_history = list()
    # def act(self, mu_vector, sigma_vector):
    #     '''
    #     input: (1, action_dim) mean vecotor
    #            (1, action_dim) standard_deviation vector
    #     output: (1, action_dim) action vector
    #     '''
    #     action_vector = torch.distributions.normal.Normal(mu_vector, sigma_vector).sample()
    #     action_vector = torch.clamp(action_vector, -1., 1.) # clipping value into the a ~ (action_space.low, action_space.high)
    #     # print()
    #     return action_vector
    # def logp(self, state, action):
        
    #     mu_vector, sigma_vector = self.forward(state)
    #     dist = torch.distributions.normal.Normal(mu_vector, sigma_vector)
    #     logp_vector = dist.log_prob(action)
    #     # print(logp_vector)
    #     logp_joint = logp_vector.sum(dim=1, keepdim=True)
    #     # print(logp_joint)
    #     # print(logp_joint)
    #     return logp_joint
    # def forward(self, x):
    #     # x = x.view([1, size])   # batch size = 1
    #     if len(self.hidden_history) > 0:
    #         h_0 = self.hidden_history[-1]
    #     else:
    #         h_0 = None

    #     x = self.rnn(x, h_0)
    #     self.hidden_history.append(x)
    #     # print(x)
    #     # print(x.shape)
    #     # exit()
    #     x = F.relu(x)
    #     x = self.out(x)
    #     if self.isTrain == False:
    #         mu_vector = x[0:self.action_dim]
    #         sigma_vector = x[self.action_dim:self.action_dim*2]
    #     else:
    #         mu_vector = x[:, 0:self.action_dim]
    #         sigma_vector = x[:, self.action_dim:self.action_dim*2]
    #     mu_vector = torch.tanh(mu_vector)
    #     # print(mu_vector)
    #     sigma_vector = F.softplus(sigma_vector)
        

    #     return mu_vector, sigma_vector
class ValueNet(nn.Module):
    def __init__(self, args):
        super(ValueNet, self).__init__()
        self.state_dim = args['state_dim']
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x