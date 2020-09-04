import gym
import gym_mouse
import time
import numpy as np
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
import os
import sys
from tensorflow.profiler.experimental import Profile
from datetime import timedelta

ENVIRONMENT = 'mouseCl-v2'

env_kwargs = dict(
    apple_num=10,
    eat_apple = 1.0,
    hit_wall = 0,
)

model_f = am.mouse_eye_brain_model

evaluate_f = tools.evaluate_mouse

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('--step', dest='total_steps',default=100000)
parser.add_argument('-n','--logname', dest='log_name',default=False)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
args = parser.parse_args()

vid_type = 'mp4'
total_steps = int(args.total_steps)
my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)


hp.epsilon = 1
hp.epsilon_min = 0.1
hp.epsilon_nstep = total_steps//2

hp.Model_save = 200000

hp.lr_start = 1e-5
hp.lr_end = 1e-10
hp.lr_nsteps = 1000000


if args.render :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
    bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
# For benchmark
st = time.time()

env = gym.make(ENVIRONMENT, **env_kwargs)
bef_o = env.reset()

if args.log_name:
    # If log directory is explicitely selected
    player = Player(
        observation_space= env.observation_space, 
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
        log_name= args.log_name
    )
else :
    player = Player(
        observation_space= env.observation_space,
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
    )
if args.render :
    env.render()

if args.profile:
    # Warm up
    for step in range(hp.Learn_start+20):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

    with Profile(f'log/{args.log_name}'):
        for step in range(5):
            action = player.act(bef_o)
            aft_o,r,d,i = env.step(action)
            player.step(bef_o,action,r,d,i)
            if d :
                bef_o = env.reset()
            else:
                bef_o = aft_o
            if args.render :
                env.render()
    remaining_steps = total_steps - hp.Learn_start - 25
    for step in range(remaining_steps):
        if ((hp.Learn_start + 25 + step) % hp.Model_save) == 0 :
            player.save_model()
            score = evaluate_f(player, gym.make(ENVIRONMENT, **env_kwargs), 
                                vid_type)
            print('eval_score:{0}'.format(score))
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(total_steps):
        if (step>0) and ((step % hp.Model_save) == 0) :
            player.save_model()
            score = evaluate_f(player, gym.make(ENVIRONMENT, **env_kwargs), 
                                vid_type)
            print('eval_score:{0}'.format(score))
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

player.save_model()
score = evaluate_f(player, gym.make(ENVIRONMENT, **env_kwargs), 
                    vid_type)
print('eval_score:{0}'.format(score))
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

