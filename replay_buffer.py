import numpy as np
import random
DISCOUNT_FACTOR = 0.99
GAE_PARAMETER = 0.95
BATCH_SIZE = 256
class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.batch_size = BATCH_SIZE
        self.current_index = 0
        self.state_buffer = np.zeros(shape=(1, self.size, 24))
        self.action_buffer = np.zeros(shape=(1, self.size, 4))
        self.reward_buffer = np.zeros(shape=(1, self.size))
        self.state_value_buffer = np.zeros(shape=(1, self.size))
        self.true_state_value_buffer = np.zeros(shape=(1, self.size))
        self.advantage_buffer = np.zeros(shape=(1, self.size))
        self.logp_buffer = np.zeros(shape=(1, self.size)) # for important sampling
        self.terminate_buffer = np.zeros(shape=(1, self.size))
        self.index_array = [x for x in range(self.size)]

        self.state_sample_batch = np.zeros(shape=(self.batch_size, 24))
        self.action_sample_batch = np.zeros(shape=(self.batch_size, 4))
        self.reward_sample_batch = np.zeros(shape=(self.batch_size))
        self.logp_sample_batch = np.zeros(shape=(self.batch_size))
        self.true_state_value_sample_batch = np.zeros(shape=(self.batch_size))
        self.advantage_sample_batch = np.zeros(shape=(self.batch_size))

    def store(self, state, action, logp, reward, state_value, isTerminate):
        self.state_buffer[0][self.current_index] = state
        self.action_buffer[0][self.current_index] = action
        self.logp_buffer[0][self.current_index] = logp
        self.reward_buffer[0][self.current_index] = reward
        self.state_value_buffer[0][self.current_index] = state_value
        self.terminate_buffer[0][self.current_index] = isTerminate

        self.current_index = (self.current_index + 1)%self.size
    def update_true_state_value(self):
        discount_factor = DISCOUNT_FACTOR
        # print(self.current_index)
        # discount_time = 0
        # value = 0
        update_index = self.size-1
        # print(update_index)
        # update_index_next = (update_index + 1) % self.size
        for i in range(self.size):
            # print(update_index)
            # exit()
            if self.terminate_buffer[0][update_index] == 0 and update_index != self.size-1: 
                self.true_state_value_buffer[0][update_index] = self.reward_buffer[0][update_index] + discount_factor * (self.true_state_value_buffer[0][update_index_next])
            else:
                self.true_state_value_buffer[0][update_index] = self.reward_buffer[0][update_index]
            update_index -= 1
            update_index_next = update_index + 1
    def update_advantage(self):
        discount_factor = DISCOUNT_FACTOR
        update_index = self.size-1
        # update_index_next = (update_index + 1) % self.size
        for i in range(self.size):
            if self.terminate_buffer[0][update_index] == 0 and update_index != self.size-1: 
                # delta_t = r(t) + GAE_PARAMETER*DISCOUNT*V()
                delta = self.reward_buffer[0][update_index] + discount_factor * (self.state_value_buffer[0][update_index_next]) - self.state_value_buffer[0][update_index]
                self.advantage_buffer[0][update_index] = delta + GAE_PARAMETER * discount_factor * (self.advantage_buffer[0][update_index_next])
            else:
                # this is an end state
                # print('hell yeah')
                # exit()
                delta = self.reward_buffer[0][update_index] - self.state_value_buffer[0][update_index]
                self.advantage_buffer[0][update_index] = delta
            update_index -= 1
            update_index_next = update_index + 1
    
    def sample(self, batch_size):
        # [state, action, reward, logp, true_state_value, advantage]
        sample_index = random.sample(self.index_array, batch_size)
        # sample_batch = np.zeros(shape=(batch_size, 6))
        sample_batch = [None for i in range(batch_size)]
        for i, element in enumerate(sample_index):
            # sample_batch[i] = [self.state_buffer[0][element], self.action_buffer[0][element], self.reward_buffer[0][element], self.logp_buffer[0][element], self.true_state_value_buffer[0][element], self.advantage_buffer[0][element]]
            self.state_sample_batch[i] = self.state_buffer[0][element]
            self.action_sample_batch[i] = self.action_buffer[0][element]
            self.reward_sample_batch[i] = self.reward_buffer[0][element]
            self.logp_sample_batch[i] = self.logp_buffer[0][element]
            self.true_state_value_sample_batch[i] = self.true_state_value_buffer[0][element]
            self.advantage_sample_batch[i] = self.advantage_buffer[0][element]
        # print(self.state_sample_batch.dtype)
        return self.state_sample_batch, self.action_sample_batch, self.reward_sample_batch, self.logp_sample_batch, self.true_state_value_sample_batch, self.advantage_sample_batch