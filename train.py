import gym
import torch
from ppo import PPO
from replay_buffer import ReplayBuffer
import wandb
BATCH_SIZE = 256
INPUT_DIM = 24
ACTION_DIM = 4
BUFFER_SIZE = 1024
EPOCHS = 3
POLICY_ARGS = {'state_dim': INPUT_DIM, 'action_dim': ACTION_DIM}
GPU = torch.cuda.is_available()
DEVICE = 'cuda' if GPU else 'cpu'
PATH = './ppo_checkpoint'
EPISODE = 1000
wandb.init(project="ppo-atari")
env = gym.make('BipedalWalker-v2').unwrapped
ppo = PPO(POLICY_ARGS)
replay_buffer = ReplayBuffer(BUFFER_SIZE)
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
    step = 0
    while True:
        if total_game_step >= BUFFER_SIZE:
            # print('gg')
            break
        state_tensor = torch.tensor([state]).float().to(DEVICE)
        mu_vector, sigma_vector = policy(state_tensor)
        action_tensor = policy.act(mu_vector, sigma_vector)
        # print(action_tensor.squeeze(0))
        next_state, reward, done, _ = env.step(action_tensor.cpu().squeeze(0).numpy())
        reward_tensor = torch.tensor([reward]).float().to(DEVICE)
        state_value = value_net(state_tensor)
        isTerminate = 1 if done else 0
        logp_tensor = policy.logp(state_tensor, action_tensor)
        # print(logp_tensor)
        state_value_tensor = value_net(state_tensor)
        replay_buffer.store(state=state_tensor.cpu().squeeze(0).numpy(), action=action_tensor.cpu().squeeze(0).numpy(), logp=logp_tensor.item(), reward=reward, state_value=state_value_tensor.item(), isTerminate=isTerminate)
        tra_reward += reward
        step += 1
        if done:
            state = env.reset()
            wandb.log({'reward': tra_reward})
            # print(step)
            step = 0
            # print(tra_reward)
            tra_reward = 0
        else:
            state = next_state
        total_game_step += 1
        # print(total_game_step)
        # exit()
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