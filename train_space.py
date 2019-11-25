
import gym
import torch
from ppo import PPO_D
from replay_buffer import ReplayBuffer
import wandb
import numpy as np
torch.cuda.empty_cache()
# BATCH_SIZE = 16
INPUT_DIM = 128
ACTION_DIM = 1
HORIZON = 2048 # = 
BATCH_SIZE = 1024
EPOCHS = 3
ENV_SIZE = 4
LEARNING_RATE = 2.5e-5
ENTROPY_COEFF = 1
isTrain = True
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './SpaceInvaders/ppo_checkpoint'
EPISODE = 1000000000
policy_args = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM, 'is_train': isTrain, 'env_size': ENV_SIZE, 'batch_size': BATCH_SIZE, 'horizon': HORIZON, 'entropy_coeff': ENTROPY_COEFF, 'learning_rate': LEARNING_RATE}

wandb.init(project="ppo-atari-SpaceInvaders", name='entropy=10_clipNorm=1')
env = gym.vector.make('SpaceInvaders-ram-v0', ENV_SIZE).unwrapped
ppo = PPO_D(policy_args)
replay_buffer = ReplayBuffer(policy_args)
# policy.load_state_dict(torch.load(PATH+ '_policy'))
# value_net.load_state_dict(torch.load(PATH+ '_valueNet'))
policy = ppo.policy
value_net = ppo.value_net
for current_episode in range(EPISODE):
    '''
    sample trajactory
    update # epoch
    '''
    total_game_step = 0
    tra_reward = 0
    # sample buffer_size steps
    state = env.reset()
    # step = 0
    average_reward = 0
    while True:
        if total_game_step >= HORIZON:
            # print('gg')
            break
        state_tensor = torch.tensor(state).float().to(DEVICE)

        action_tensor = policy.act(state_tensor).detach()
        next_state, reward, done, _ = env.step(tuple(action_tensor.cpu().numpy()))
        
        logp_tensor = policy.logp(state_tensor, action_tensor).detach()

        next_state_tensor = torch.tensor(next_state).float().to(DEVICE)
        state_value_tensor = value_net(state_tensor).detach()
        next_state_value_tensor = value_net(next_state_tensor).detach()
        average_reward += np.mean(reward)
        replay_buffer.store(state=state_tensor.cpu().numpy(), next_state=next_state, action=action_tensor.cpu().numpy(), logp=logp_tensor.cpu().numpy(), reward=reward, state_value=state_value_tensor.cpu().numpy(), next_state_value=next_state_value_tensor.cpu().numpy() , isTerminate=done)
        
        state = next_state
        total_game_step += 1
        # print(total_game_step)
        
    # sample finished
    wandb.log({'average_reward': average_reward})
    replay_buffer.update_true_state_value()
    replay_buffer.update_advantage()
    replay_buffer.merge_trajectory()
    for epoch in range(EPOCHS):
        actor_loss, critic_loss =  ppo.update_discrete(replay_buffer)
        wandb.log({'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()})
        if current_episode % 10 == 0:
            print(policy(state_tensor).detach())
            print('-------------------------')
            print('episode: {episode}, actor_loss: {actor_loss}, critic_losss: {critic_losss}' \
                .format(episode=current_episode, actor_loss=actor_loss, critic_losss=critic_loss))
            torch.save(ppo.policy.state_dict(), PATH + '_policy')
            torch.save(ppo.value_net.state_dict(), PATH + '_valueNet')

env.close() 