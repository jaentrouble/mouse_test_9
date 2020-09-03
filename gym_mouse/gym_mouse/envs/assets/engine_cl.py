from __future__ import absolute_import
import numpy as np
from .managers import *
from .things.static_things import Apple
from .things.dynamic_things import Mouse
from .things.things_consts import DefaultSize as ds
from ..constants import colors, rng
# from ..constants import rewards as R
from ..constants import engine_const as ec
# installation mistake
import pyopencl as cl
import pyopencl.array as cl_array
from os import path

#**** When drawing things on a grid or an image, do it in the order of id.
#       This is to get consistent across engine and collision manager.

class Engine():
    """
    Game engine that calculates all interactions
    Image is the RGB array
    Grid is the array that contains id number of all things
    """
    def __init__(self, size, **kwargs) :
        """
        Parameters
        ----------
        size : tuple of two int
            (height, width) of the map

        kwargs
        ------
        apple_num : int
            number of total apples in a map
        eat_apple : float
            reward given when apple is eaten.
        hit_wall : float
            punishment(or reward?) given when hit wall.
        """
        # Don't confuse 'Viewer' and 'Engine'

        # kwargs
        self._apple_num = kwargs['apple_num']
        self._rewards = dict(
            eat_apple = kwargs['eat_apple'],
            hit_wall = kwargs['hit_wall'],
        )

        # Size of Engine should always be the same while running
        self._height = size[0]
        self._width = size[1]
        self._image = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        self._TM = ThingsManager()

        # OpenCl things
        self.device = cl.get_platforms()[0].get_devices()[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.bg_color = np.array(colors.COLOR_BACKGROUND, dtype=np.uint8)
        self.wall_color = np.array(colors.COLOR_WALL, dtype=np.uint8)
        self.image_dev = cl_array.empty(self.queue,self.image.shape,np.uint8)
        self.bg_col_dev = cl_array.to_device(self.queue, self.bg_color)
        self.wall_col_dev = cl_array.to_device(self.queue, self.wall_color)
        self.fp_ray_dev = None
        self.delta_vec_dev = None
        self.observation_dev = cl_array.empty(self.queue,(2,ec.RayNum,3),np.uint8)
        cl_path = path.join(path.dirname(__file__),'cl_scripts/ray.cl')
        with open(cl_path,'r') as f:
            fstr = "".join(f.readlines())
        self.program = cl.Program(self.ctx, fstr).build()

        # Initiate things first and then call CollisionManager
        self.initiate_things()
        self._CM = CollisionManager(self.size, self._TM)

    @property
    def size(self):
        return  (self._height, self._width)

    @property
    def image(self):
        return self._image.copy()

    def initial_observation(self):
        """
        Returns current observation
        Use this for initial observation
        """
        return {'Right':np.array(self._obs_rt_cache,
                        dtype=np.uint8).swapaxes(0,1),
                'Left':np.array(self._obs_lt_cache,
                        dtype=np.uint8).swapaxes(0,1)}
    def initiate_things(self):
        """
        Initiate and register things to thingsmanager
        Recommand to register mouse very first.
        """
        min_r, min_c = ds.Mouse_max_len, ds.Mouse_max_len
        max_r = self.size[0] - ds.Mouse_max_len
        max_c = self.size[1] - ds.Mouse_max_len

        self.apples = []
        for _ in range(self._apple_num):
            rand_a_pos = (rng.np_random.randint(0,self.size[0]),
                        rng.np_random.randint(0,self.size[1]))
            apple = Apple(rand_a_pos, self.size)
            self._TM.regist(apple)
            self.apples.append(apple)

        rand_m_pos = (rng.np_random.randint(min_r,max_r),
                      rng.np_random.randint(min_c,max_c))
        self.The_mouse = Mouse(
            rand_m_pos,
            rng.np_random.rand()*np.pi, 
            self.size,
            self._rewards
        )
        self._mouse_ID = self._TM.regist(self.The_mouse)

        for color, idx in self._TM.all_color:
            self._image[idx[0],idx[1]] = color
        lt_obs, rt_obs = self.observe()
        self._obs_lt_cache = []
        self._obs_rt_cache = []

        for _ in range(ec.CacheNum):
            self._obs_lt_cache.append(lt_obs)
            self._obs_rt_cache.append(rt_obs)

    def update(self, action):
        """
        action : (Delta_center, Delta_theta)
        """
        # Reset first, so that static things will not have problem when
        # they are created at the edge.
        # To keep track of scores(How many apples did it manage to get)
        info = {'ate_apple':False}
        self._TM.reset_updated()
        mouse_reward, done, ate_apple = self._CM.update(action, self._mouse_ID)
        if ate_apple:
            info['ate_apple']=True
        reward = self.reward_calc(mouse_reward)
        for color, updated_idx, last_idx in self._TM.updated_color:
            self._image[last_idx[0],last_idx[1]] = colors.COLOR_BACKGROUND
            self._image[updated_idx[0],updated_idx[1]] = color
        lt_obs, rt_obs = self.observe()
        self._obs_lt_cache.pop(0)
        self._obs_rt_cache.pop(0)
        self._obs_lt_cache.append(lt_obs)
        self._obs_rt_cache.append(rt_obs)
        observation = {'Right':np.array(self._obs_rt_cache,
                            dtype=np.uint8).swapaxes(0,1),
                        'Left':np.array(self._obs_lt_cache,
                            dtype=np.uint8).swapaxes(0,1)}
        # Last axis has RGB values
        return observation, reward, done, info

    def observe(self):
        """
        return lt_obs, rt_obs
        """
        lt_eye, rt_eye, theta, beta = self.The_mouse.eye
        # Offset
        lt_eye = np.round(lt_eye + 1.5* np.array([np.cos(theta+beta),
                                                np.sin(theta+beta)]))[:,np.newaxis]
        rt_eye = np.round(rt_eye + 1.5* np.array([np.cos(theta-beta),
                                                np.sin(theta-beta)]))[:,np.newaxis]
        fp_ray = np.stack((np.broadcast_to(lt_eye,(2,ec.RayNum)),
                        np.broadcast_to(rt_eye,(2,ec.RayNum))),
                        axis=0).astype(np.float32)
        lt_angles = np.linspace(theta+beta+np.pi/2, theta+beta-np.pi/2,num=ec.RayNum)
        rt_angles = np.linspace(theta-beta-np.pi/2, theta-beta+np.pi/2,num=ec.RayNum)

        delta_vec = (np.stack((np.cos(lt_angles),np.sin(lt_angles),
                              np.cos(rt_angles),np.sin(rt_angles)),
                              axis=0)*2).astype(np.float32)
        delta_vec.resize(2,2,ec.RayNum)
        if self.fp_ray_dev is None :
            self.fp_ray_dev = cl_array.to_device(self.queue, fp_ray)
            self.delta_vec_dev = cl_array.to_device(self.queue, delta_vec)
        else:
            self.fp_ray_dev.set(fp_ray)
            self.delta_vec_dev.set(delta_vec)
        self.image_dev.set(self.image)
        args = (
            self.image_dev.data,
            np.int32(self.size[0]),
            np.int32(self.size[1]),
            np.int32(ec.RayNum),
            np.int32(ec.LightRatio),
            self.bg_col_dev.data,
            self.wall_col_dev.data,
            self.fp_ray_dev.data,
            self.delta_vec_dev.data,
            self.observation_dev.data
        )
        self.program.ray(self.queue, (2,ec.RayNum), None,*args)
        lt_obs, rt_obs = self.observation_dev.get()
        return lt_obs, rt_obs
    

    def reward_calc(self, mouse_reward):
        """
        Reward calculation function
        To add something other than mouse's reward
        """
        reward = mouse_reward
        # new_dist = np.sqrt(np.sum(np.subtract(self.The_apple.pos,
        #                                       self.The_mouse.nose)**2))
        # # If mouse gets farther away from the apple, punish
        # if new_dist > self._a_m_dist:
        #     reward += R.get_away_from_apple
        # elif new_dist < self._a_m_dist:
        #     reward += R.get_close_to_apple
        # self._a_m_dist = new_dist
        return reward
        