import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions
from network import Network, ValueNet
import wandb
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
BATCH_SIZE = 512
INPUT_DIM = 24
ACTION_DIM = 4
# BUFFER_SIZE = 1024
isTrain = True
EPOCHS = 3
EPISLON = 0.2
LEARNING_RATE = 2.5e-4
VF_COEFF = 1
ENTROPY_COEFF = 0.01
ENV_SIZE = 8
# WEIGHT_DECAY = 0.99
# MOMENTUM = 0.9

class PPO(object):
    def __init__(self, args):
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.batch_size = args['batch_size']
        self.network_args = {'state_dim': self.state_dim, 'action_dim': self.action_dim, 'is_train': args['is_train']}
        # network_args = {'state_dim': self.state_dim, 'action_dim': self.action_dim}
        self.policy = Network(self.network_args).to(DEVICE)
        self.value_net = ValueNet(self.network_args).to(DEVICE)
        self.mse = nn.MSELoss()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        # self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
    def update(self, replay_buffer):
        '''
        update policy by sample from replay buffer 
        '''
        state_array, action_array, reward_array, logp_old_array, true_state_value_array, advantage_array = replay_buffer.sample(self.batch_size)
        
        state_batch = torch.tensor(state_array).float().to(DEVICE)
        action_batch = torch.tensor(action_array).float().to(DEVICE)
        # reward_batch = torch.tensor(reward_array).float().to(DEVICE)
        logp_old_batch = torch.tensor(logp_old_array).float().to(DEVICE)
        true_state_value_batch = torch.tensor(true_state_value_array).float().to(DEVICE)
        advantage_batch = torch.tensor(advantage_array).float().to(DEVICE)
        
        state_value_batch = self.value_net(state_batch)
        logp_batch = self.policy.logp(state_batch, action_batch)
        
        logp_old_batch = logp_old_batch.unsqueeze(1)
        true_state_value_batch = true_state_value_batch.unsqueeze(1)
        advantage_batch = advantage_batch.unsqueeze(1)
        # print(logp_old_batch.shape)
        # print(logp_batch.shape)
        ratio_batch = torch.exp(logp_batch - logp_old_batch) # A/B = exp(logA - logB)
        ratio_clip_batch = torch.clamp(ratio_batch, 1 - EPISLON, 1 + EPISLON)
        # print(logp_batch.shape)
        # print(logp_old_batch.shape)
        # define loss
        # print(logp_old_batch.shape)
        
        # print(logp_batch.shape)
        value_function_loss = self.mse(state_value_batch, true_state_value_batch)
        clip_loss = -torch.mean(torch.min(ratio_batch * advantage_batch, ratio_clip_batch * advantage_batch))
        # print(ratio_batch.shape)
        # print(advantage_batch.shape)
        # print(clip_loss)
        # mu_vector_batch, sigma_vector_batch = self.policy(state_batch)
        # state_entropy_batch = torch.mean(torch.distributions.normal.Normal(mu_vector_batch, sigma_vector_batch).entropy())
        # print(true_state_value_batch)
        state_entropy_batch = -logp_batch*torch.exp(logp_batch)
        entropy_loss = -torch.mean(state_entropy_batch)
        actor_loss = clip_loss + ENTROPY_COEFF*entropy_loss
        # print(actor_loss)
        # print(entropy_loss)
        critic_loss = VF_COEFF*value_function_loss
        wandb.log({'clip_loss': clip_loss.item(), 'entropy_loss': ENTROPY_COEFF*entropy_loss.item()})

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        self.value_net_optimizer.zero_grad()
        critic_loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.value_net_optimizer.step()
        return actor_loss, critic_loss
        
    def sample(buffer_size):
        '''
        sample buffer_size trajactory into buffer
        '''


