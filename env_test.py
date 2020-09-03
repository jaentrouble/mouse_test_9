import gym
import gym_mouse
import time
from tqdm import trange
import numpy as np

EYE = True

env_kwargs = dict(
    apple_num=50,
    eat_apple = 1.0,
    hit_wall = -1.0,
)


if EYE:
    from gym.envs.classic_control import rendering
    eye_viewer = rendering.SimpleImageViewer(maxwidth=1100)
    eye_bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)

st = time.time()
env = gym.make('mouseCl-v2', **env_kwargs)
env.seed(3)
env.reset()
# diff = 0
# for _ in trange(100):
    # diff += env.check_step(env.action_space.sample())
for _ in trange(100):
    o, r, d, i = env.step(env.action_space.sample())
    if EYE:
        rt_eye = np.flip(o['Right'][:,-1,:],axis=0)
        lt_eye = o['Left'][:,-1,:]
        eye_img = np.concatenate((lt_eye,eye_bar,rt_eye))
        eye_img = np.broadcast_to(eye_img.reshape((1,205,1,3)),(50,205,5,3))
        eye_img = eye_img.reshape(50,205*5,3)
        eye_viewer.imshow(eye_img)
        env.render()
        time.sleep(0.1)
    if d :
        env.reset()
# env.render()
    # env.render()
# print(diff)
# input('done:')
print(time.time() - st)