import gym
# env = gym.vector.make('BipedalWalker-v2', 3)
env = gym.make('BipedalWalker-v2')
# env = gym.make('SpaceInvaders-ram-v0')
env.reset()

for _ in range(200000):
    # env.render()
    action = env.action_space.sample()
    print(env.action_space.low)
    state, reward, done, info = env.step([float('nan'), 2., 2., 2.])
    # print(action)
    # print(type(action))
    # exit()
    # print(state.shape)
    # if done[0] == True or done[1] == True or done[2] == True:
    #     print(done)
        # break
    # print(reward)
env.close()