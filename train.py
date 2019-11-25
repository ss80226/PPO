


from gym_unity.envs import UnityEnv
from replay_buffer import ReplayBuffer
import torch
import numpy as np 
from ppo import PPO
import wandb

wandb.init(project='ppo-unity-walker', name='entropy=0.01_clipNorm=1')
torch.cuda.empty_cache()
INPUT_DIM = 212
ACTION_DIM = 39
HORIZON = 1024 # = 
BATCH_SIZE = 1024
EPOCHS = 3
ENV_SIZE = 11
LEARNING_RATE = 2.5e-4
ENTROPY_COEFF = 0.001
isTrain = True
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './unityWalker/ppo_checkpoint'
EPISODE = 1000000000
policy_args = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM, 'is_train': isTrain, 'env_size': ENV_SIZE, 'batch_size': BATCH_SIZE, 'horizon': HORIZON, 'entropy_coeff': ENTROPY_COEFF, 'learning_rate': LEARNING_RATE}

replay_buffer = ReplayBuffer(args=policy_args)
ppo = PPO(args=policy_args)
policy = ppo.policy
value_net = ppo.value_net
# policy.load_state_dict(torch.load(PATH+ '_policy'))
# value_net.load_state_dict(torch.load(PATH+ '_valueNet'))

env = UnityEnv('walker', multiagent=True, worker_id = 11).unwrapped

for current_episode in range(EPISODE):
    # env.render()
    env.reset()
    state_list = env.reset()
    average_reward = 0
    total_game_step = 0
    while True:
        if total_game_step >= HORIZON:
            break
        state_tensor = torch.tensor(state_list).float().to(DEVICE)
        mu_vector, sigma_vector = policy(state_tensor)
        action_tensor = policy.act(mu_vector, sigma_vector)
        action_list = []
        for action in action_tensor.cpu().numpy():
            action_list.append(action)

        next_state_list, reward_list, done_list, _ = env.step(action_list)
        logp_tensor = policy.logp(state_tensor, action_tensor).detach()
        state_value_tensor = value_net(state_tensor).detach()
        next_state_tensor = torch.tensor(next_state_list).float().to(DEVICE)
        next_state_value_tensor = value_net(next_state_tensor).detach()

        replay_buffer.store(state=state_tensor.cpu().numpy(), action=action_tensor.cpu().numpy(), reward=reward_list, state_value=state_value_tensor.cpu().numpy(), isTerminate=done_list, next_state=next_state_list, logp=logp_tensor.cpu().numpy(), next_state_value=next_state_value_tensor.cpu().numpy())
        state_list = next_state_list
        total_game_step += 1
        average_reward += sum(reward_list) / len(reward_list)        
        
    replay_buffer.update_true_state_value()
    replay_buffer.update_advantage()
    replay_buffer.merge_trajectory()
    actor_loss, critic_loss =  ppo.update(replay_buffer)
    wandb.log({'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'average_reward': average_reward})
    if current_episode % 100 == 0:
        print('-------------------------')
        print('episode: {episode}, actor_loss: {actor_loss}, critic_losss: {critic_losss}\n average_reward: {average_reward}' \
            .format(episode=current_episode, actor_loss=actor_loss, critic_losss=critic_loss, average_reward=average_reward))
        torch.save(ppo.policy.state_dict(), PATH + '_policy')
        torch.save(ppo.value_net.state_dict(), PATH + '_valueNet')
        print()

env.close()