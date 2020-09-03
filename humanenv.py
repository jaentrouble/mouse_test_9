import gym
import gym_mouse

env_kwargs = dict(
    apple_num=10,
    eat_apple = 1.0,
    hit_wall = -1.0,
)

env = gym.make('mouseCl-v2', **env_kwargs)
env.seed(3)
env.reset()
total_reward = 0
while True :
    env.render()
    a = int(input('Move:'))
    if a == -1 :
        break
    o, r, d, i = env.step(a)
    total_reward += r
    print('reward : {}'.format(r))
    if d :
        env.reset()
        print('done, total reward:{}'.format(total_reward))
        total_reward = 0
