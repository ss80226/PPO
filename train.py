import gym
import torch
from ppo import PPO
from replay_buffer import ReplayBuffer
import wandb
torch.cuda.empty_cache()
BATCH_SIZE = 16
INPUT_DIM = 24
ACTION_DIM = 4
HORIZON = 128 # = 
BUFFER_SIZE = 1024
EPOCHS = 3
ENV_SIZE = 8
POLICY_ARGS = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM}
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './ppo_checkpoint'
EPISODE = 1000000000
wandb.init(project="ppo-atari")
env = gym.vector.make('BipedalWalker-v2', ENV_SIZE).unwrapped
ppo = PPO(POLICY_ARGS)
replay_buffer = ReplayBuffer(HORIZON)
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
    while True:
        if total_game_step >= HORIZON:
            # print('gg')
            break
        state_tensor = torch.tensor(state).float().to(DEVICE)
        mu_vector, sigma_vector = policy(state_tensor)
        # print(mu_vector)
        action_tensor = policy.act(mu_vector, sigma_vector)
        # print(action_tensor.squeeze(0))
        # print(mu_vector)
        # exit()
        next_state, reward, done, _ = env.step(tuple(action_tensor.cpu().numpy()))
        # reward_tensor = torch.tensor([reward]).float().to(DEVICE)
        # state_value = value_net(state_tensor)
        
        logp_tensor = policy.logp(state_tensor, action_tensor).detach()
        # print(logp_tensor)
        # exit()
        # print(logp_tensor)
        state_value_tensor = value_net(state_tensor).detach()
        # print(reward)
        # print(reward)
        # print(done)
        # exit()
        # print(state_value_tensor)
        # exit()
        replay_buffer.store(state=state_tensor.cpu().numpy(), action=action_tensor.cpu().numpy(), logp=logp_tensor.cpu().numpy(), reward=reward, state_value=state_value_tensor.cpu().numpy(), isTerminate=done)
        # tra_reward += reward
        # step += 1
        # if done:
        #     # print('qqq')
        #     state = env.reset()
        #     # wandb.log({'reward': tra_reward})
        #     # print(step)
        #     step = 0
        #     # print(tra_reward)
        #     tra_reward = 0
        # else:
        state = next_state
        total_game_step += 1
        # print(total_game_step)
        # exit()
    # sample finished
    replay_buffer.update_true_state_value()
    replay_buffer.update_advantage()
    for epoch in range(EPOCHS):
        actor_loss, critic_loss =  ppo.update(replay_buffer)
        wandb.log({'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()})
        if current_episode % 10 == 0:
            print('-------------------------')
            print('episode: {episode}, actor_loss: {actor_loss}, critic_losss: {critic_losss}' \
                .format(episode=current_episode, actor_loss=actor_loss, critic_losss=critic_loss))
            torch.save(ppo.policy.state_dict(), PATH + '_policy')
            torch.save(ppo.value_net.state_dict(), PATH + '_valueNet')

env.close() 