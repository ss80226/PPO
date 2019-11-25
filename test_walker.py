import gym
import torch
from ppo import PPO
from replay_buffer import ReplayBuffer
import wandb
import numpy as np
import time
torch.cuda.empty_cache()
# BATCH_SIZE = 16
INPUT_DIM = 24
ACTION_DIM = 4
HORIZON = 2048 # = 
BATCH_SIZE = 256
EPOCHS = 3
ENV_SIZE = 4
LEARNING_RATE = 2.5e-4
ENTROPY_COEFF = 1
isTrain = False
policy_args = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM, 'is_train': isTrain, 'env_size': ENV_SIZE, 'batch_size': BATCH_SIZE, 'horizon': HORIZON, 'entropy_coeff': ENTROPY_COEFF, 'learning_rate': LEARNING_RATE}
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './BipedalWalker/ppo_checkpoint2'
EPISODE = 1000000000
# wandb.init(project="ppo-BipedalWalker")
# env = gym.vector.make('BipedalWalker-v2', ENV_SIZE).unwrapped
ppo = PPO(policy_args)
replay_buffer = ReplayBuffer(policy_args)
policy = ppo.policy
value_net = ppo.value_net
policy.load_state_dict(torch.load(PATH+ '_policy'))
value_net.load_state_dict(torch.load(PATH+ '_valueNet'))

env = gym.make('BipedalWalker-v2')
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
        print(reward)
        state_value_tensor = value_net(state_tensor).detach()
        
        # replay_buffer.store(state=state_tensor.cpu().numpy(), action=action_tensor.cpu().numpy(), reward=reward, state_value=state_value_tensor.cpu().numpy(), isTerminate=done)
        state_list = next_state_list
        total_game_step += 1
        average_reward += np.mean(reward)
        if done:
            print(total_game_step)
            break
        
    # replay_buffer.update_true_state_value()
    # replay_buffer.update_advantage()
    # replay_buffer.merge_trajectory()
    # actor_loss, critic_loss =  a2c.update(replay_buffer)
    # wandb.log({'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'average_reward': average_reward})
    # if current_episode % 100 == 0:
    #     print('-------------------------')
    #     print('episode: {episode}, actor_loss: {actor_loss}, critic_losss: {critic_losss}' \
    #         .format(episode=current_episode, actor_loss=actor_loss, critic_losss=critic_loss))
    #     torch.save(a2c.policy.state_dict(), PATH + '_policy')
    #     torch.save(a2c.value_net.state_dict(), PATH + '_valueNet')
    # print(average_reward)

env.close()