import numpy as np
import random
DISCOUNT_FACTOR = 0.99
GAE_PARAMETER = 0.95
BATCH_SIZE = 256
ENV_SIZE = 8
class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.batch_size = BATCH_SIZE
        self.env_size = ENV_SIZE
        self.current_index = 0
        self.state_buffer = np.zeros(shape=(self.env_size, self.size, 24))
        self.action_buffer = np.zeros(shape=(self.env_size, self.size, 4))
        self.reward_buffer = np.zeros(shape=(self.env_size, self.size))
        self.state_value_buffer = np.zeros(shape=(self.env_size, self.size))
        self.true_state_value_buffer = np.zeros(shape=(self.env_size, self.size))
        self.advantage_buffer = np.zeros(shape=(self.env_size, self.size))
        self.logp_buffer = np.zeros(shape=(self.env_size, self.size)) # for important sampling
        self.terminate_buffer = np.zeros(shape=(self.env_size, self.size))
        self.index_array = [x for x in range(self.size)]

        self.state_sample_batch = np.zeros(shape=(self.batch_size, 24))
        self.action_sample_batch = np.zeros(shape=(self.batch_size, 4))
        self.reward_sample_batch = np.zeros(shape=(self.batch_size))
        self.logp_sample_batch = np.zeros(shape=(self.batch_size*self.env_size))
        self.true_state_value_sample_batch = np.zeros(shape=(self.batch_size))
        self.advantage_sample_batch = np.zeros(shape=(self.batch_size))

    def store(self, state, action, logp, reward, state_value, isTerminate):
        for i in range(self.env_size):
            self.state_buffer[i][self.current_index] = state[i]
            self.action_buffer[i][self.current_index] = action[i]
            self.logp_buffer[i][self.current_index] = logp[i]
            self.reward_buffer[i][self.current_index] = reward[i]
            self.state_value_buffer[i][self.current_index] = state_value[i]
            self.terminate_buffer[i][self.current_index] = 1 if isTerminate[i] else 0

        self.current_index = (self.current_index + 1)%self.size
    def update_true_state_value(self):
        # print(self.current_index)
        discount_factor = DISCOUNT_FACTOR
        # print(self.current_index)
        # discount_time = 0
        # value = 0
        # self.true_state_value_buffer = np.zeros(shape=(1, self.size))
        
        # print(update_index)
        # update_index_next = (update_index + 1) % self.size
        for i in range(self.env_size):
            update_index = self.size-1
            for j in range(self.size):
                # print(update_index)
                # exit()
                if self.terminate_buffer[i][update_index] == 0 and update_index != self.size-1: 
                    self.true_state_value_buffer[i][update_index] = self.reward_buffer[i][update_index] + discount_factor * (self.true_state_value_buffer[i][update_index_next])
                else:
                    self.true_state_value_buffer[i][update_index] = self.reward_buffer[i][update_index]
                    # print('qq')
                # print(self.true_state_value_buffer[0][update_index])
                update_index -= 1
                update_index_next = update_index + 1
        # print(self.true_state_value_buffer)
    def update_advantage(self):
        discount_factor = DISCOUNT_FACTOR
        
        # update_index_next = (update_index + 1) % self.size
        for i in range(self.env_size):
            update_index = self.size-1
            for j in range(self.size):
                if self.terminate_buffer[i][update_index] == 0 and update_index != self.size-1: 
                    # delta_t = r(t) + GAE_PARAMETER*DISCOUNT*V()
                    delta = self.reward_buffer[i][update_index] + discount_factor * (self.state_value_buffer[i][update_index_next]) - self.state_value_buffer[i][update_index]
                    self.advantage_buffer[i][update_index] = delta + GAE_PARAMETER * discount_factor * (self.advantage_buffer[i][update_index_next])
                else:
                    # this is an end state
                    # print('hell yeah')
                    # exit()
                    delta = self.reward_buffer[i][update_index] - self.state_value_buffer[i][update_index]
                    self.advantage_buffer[i][update_index] = delta
                update_index -= 1
                update_index_next = update_index + 1
    
    def sample(self, batch_size):
        # [state, action, reward, logp, true_state_value, advantage]
        index = int(batch_size / self.env_size)
        sample_index = random.sample(self.index_array, index)
        # sample_batch = np.zeros(shape=(batch_size, 6))
        for j in range(self.env_size):
            for i, element in enumerate(sample_index):
                # sample_batch[i] = [self.state_buffer[0][element], self.action_buffer[0][element], self.reward_buffer[0][element], self.logp_buffer[0][element], self.true_state_value_buffer[0][element], self.advantage_buffer[0][element]]
                self.state_sample_batch[i+j*index] = self.state_buffer[j][element]
                self.action_sample_batch[i+j*index] = self.action_buffer[j][element]
                self.reward_sample_batch[i+j*index] = self.reward_buffer[j][element]
                self.logp_sample_batch[i+j*index] = self.logp_buffer[j][element]
                self.true_state_value_sample_batch[i+j*index] = self.true_state_value_buffer[j][element]
                self.advantage_sample_batch[i+j*index] = self.advantage_buffer[j][element]
        # print(self.state_sample_batch.dtype)
        return self.state_sample_batch, self.action_sample_batch, self.reward_sample_batch, self.logp_sample_batch, self.true_state_value_sample_batch, self.advantage_sample_batch