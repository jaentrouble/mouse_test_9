from sanity_env import EnvTest
import gym
import gym_mouse
import numpy as np
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
import os
import sys
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('-pf', dest='profile', action='store_true', default=False)
parser.add_argument('-sa', dest='sanity_agent', action='store_true',default=False)
parser.add_argument('--steps', dest='steps')
parser.add_argument('--logname', dest='log_name', default=None)
args = parser.parse_args()

total_steps = int(args.steps)

my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)

from CartPoleAgent import Player

hp.Buffer_size = 1000
hp.Learn_start = 200
hp.Batch_size = 32
hp.Target_update = 500
hp.epsilon = 1
hp.epsilon_min = 0.01
hp.epsilon_nstep = 500

original_env = gym.make('CartPole-v1')
test_env = EnvTest(original_env.observation_space)
player = Player(original_env.observation_space, test_env.action_space, my_tqdm,
                log_name=args.log_name)
bef_o = test_env.reset()
# for step in trange(1000) :
#     player.act(o,training=True)
#     if step%5 == 0 :
#         action = 2
#     elif step%5 == 1 :
#         action = 1
#     elif step%5 == 2 :
#         action = 2
#     elif step%5 == 3 :
#         action = 1
#     elif step%5 == 4 :
#         action = 0
#     o, r, d, i = test_env.step(action)
#     player.step(action, r,d,i)
#     if d :
#         o = test_env.reset()
if args.profile:
    for step in range(hp.Learn_start+50):
        action = player.act(bef_o)
        aft_o, r, d, i = test_env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = test_env.reset()
        else :
            bef_o = aft_o
    with tf.profiler.experimental.Profile('log/profile'):
        for step in range(5):
            action = player.act(bef_o)
            aft_o, r, d, i = test_env.step(action)
            player.step(bef_o,action,r,d,i)
            if d :
                bef_o = test_env.reset()
            else :
                bef_o = aft_o

else :
    for step in range(int(args.steps)):
        action = player.act(bef_o)
        aft_o, r, d, i = test_env.step(action)
        player.step(bef_o, action,r,d,i)
        if d :
            bef_o = test_env.reset()
        else :
            bef_o = aft_o
        # if step % 1000 == 0 :
        #     print('Evaluating')
        #     vo = test_env.reset()
        #     rewards = 0
        #     for _ in trange(50):
        #         vaction = player.act(vo, training=False)
        #         print(vaction)
        #         vo, vr, vd, vi = test_env.step(vaction)
        #         print(vr)
        #         rewards += vr
        #         if vd :
        #             vo = test_env.reset()
        #     print(rewards/10)
        #     input('continue?')
my_tqdm.close()
