import gym
env = gym.make('BipedalWalker-v2')
# env = gym.make('CartPole-v0')
# env = gym.make('SpaceInvaders-ram-v0')
env.reset()

for _ in range(200000):
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(state.shape)
    if done:
        print('done')
        break
    # print(reward)
env.close()