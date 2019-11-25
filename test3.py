import gym
import torch
from ppo import PPO
from replay_buffer import ReplayBuffer
import wandb
import numpy as np
import time
torch.cuda.empty_cache()
# BATCH_SIZE = 16
INPUT_DIM = 8
ACTION_DIM = 2
HORIZON = 2048 # = 
BATCH_SIZE = 256
EPOCHS = 3
ENV_SIZE = 4
LEARNING_RATE = 2.5e-4
ENTROPY_COEFF = 0.01
isTrain = False
policy_args = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM, 'is_train': isTrain, 'env_size': ENV_SIZE, 'batch_size': BATCH_SIZE, 'horizon': HORIZON, 'entropy_coeff': ENTROPY_COEFF, 'learning_rate': LEARNING_RATE}
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './LunarLanderContinuous/ppo_checkpoint2'
EPISODE = 1000000000
# wandb.init(project="ppo-BipedalWalker")
# env = gym.vector.make('BipedalWalker-v2', ENV_SIZE).unwrapped
ppo = PPO(policy_args)
replay_buffer = ReplayBuffer(policy_args)
policy = ppo.policy
value_net = ppo.value_net
policy.load_state_dict(torch.load(PATH+ '_policy'))
value_net.load_state_dict(torch.load(PATH+ '_valueNet'))

env = gym.make('LunarLanderContinuous-v2')
for current_episode in range(10):
    
    # env.reset()
    state_list = env.reset()
    average_reward = 0
    total_game_step = 0
    while True:
        # if total_game_step >= HORIZON:
        #     break
        env.render()
        time.sleep(0.01)    
        state_tensor = torch.tensor(state_list).float().to(DEVICE)
        mu_vector, sigma_vector = policy(state_tensor)
        action_tensor = policy.act(mu_vector, sigma_vector)
        
        next_state_list, reward, done, _ = env.step(action_tensor.cpu().numpy())
        
        state_value_tensor = value_net(state_tensor).detach()
        
        # replay_buffer.store(state=state_tensor.cpu().numpy(), action=action_tensor.cpu().numpy(), reward=reward, state_value=state_value_tensor.cpu().numpy(), isTerminate=done)
        state_list = next_state_list
        total_game_step += 1
        average_reward += np.mean(reward)
        if done:
            print(total_game_step)
            break
    

env.close()