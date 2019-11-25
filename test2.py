import gym
import torch
from ppo import PPO_D
from replay_buffer import ReplayBuffer
import wandb
import numpy as np
import time
torch.cuda.empty_cache()
# BATCH_SIZE = 16
INPUT_DIM = 128
ACTION_DIM = 1
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
PATH = './SpaceInvaders/ppo_checkpoint'
EPISODE = 1000000000
# wandb.init(project="ppo-BipedalWalker")
# env = gym.vector.make('BipedalWalker-v2', ENV_SIZE).unwrapped
ppo = PPO_D(policy_args)
replay_buffer = ReplayBuffer(policy_args)
policy = ppo.policy.eval()
value_net = ppo.value_net.eval()
policy.load_state_dict(torch.load(PATH+ '_policy'))
value_net.load_state_dict(torch.load(PATH+ '_valueNet'))

env = gym.make('SpaceInvaders-ram-v0')
for current_episode in range(10):
    
    # env.reset()
    state = env.reset()
    average_reward = 0
    total_game_step = 0
    while True:
        env.render()
        time.sleep(0.05)

        state_tensor = torch.tensor(state).float().to(DEVICE)

        action_tensor = policy.act(state_tensor)
        # if action_tensor.cpu().numpy() != 4:
        #     print(action_tensor.cpu().numpy())
        next_state, reward, done, _ = env.step(action_tensor.cpu().numpy())
        print(reward)
        average_reward += np.mean(reward)
        
        state = next_state
        total_game_step += 1
        if done:
            print(total_game_step)
            break
    

env.close()