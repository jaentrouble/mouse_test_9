from .base_things import Base
from ...constants import colors, tools
import numpy as np
from skimage import draw
from .things_consts import ThingsType as tt
from .things_consts import DefaultSize as ds


class Mouse(Base):
    """
    Mouse
    Our Hero
    """
    def __init__(self, center, theta, shape, rewards):
        """
        Arguments
        ---------
        center : tuple of two int
            Center coordinate of the mouse
        theta : float
            The direction the mouse is heading, in radian
        shape : tuple of two int
            Maximum size of the grid

        rewards : dict
            eat_apple : float
                Reward when eat apple
            hit_wall : float
                Punishment when hit wall
        """
        super().__init__()
        self._half_width = ds.Mouse_half_width
        self._half_height = ds.Mouse_half_height
        self._nose_len = 5
        self._shape = shape
        # Angle between axis of the body and the 'eyes'
        self._alpha = np.tanh(self._half_width/self._half_height)
        # Angle between axis of the body and eye-to-nose
        self._beta = np.tanh(self._nose_len/self._half_width)
        self._R = np.sqrt(self._half_height**2 + self._half_width**2)
        self.update_pos(center, theta)
        self.color = colors.COLOR_MOUSE
        self._t_type = tt.Mouse
        self._reward = 0
        self._dead = False
        self._ate_apple = False
        self._reward_dict = rewards

    def update_pos(self, center, theta):
        self._center = np.array(center)
        self._theta = theta
        self._nose_pos = center + \
                        (self._half_height + self._nose_len) * \
                        np.array((np.cos(theta),np.sin(theta)))
        self._rt_f = center + self._R * np.array((np.cos(theta-self._alpha),
                                                np.sin(theta-self._alpha)))
        self._lt_f = center + self._R * np.array((np.cos(theta+self._alpha),
                                                np.sin(theta+self._alpha)))
        self._lt_b = center - (self._rt_f - center)
        self._rt_b = center - (self._lt_f - center)
        stacked = np.stack((self._rt_f,
                            self._nose_pos,
                            self._lt_f,
                            self._lt_b,
                            self._rt_b), axis=1)
        self.indices = draw.polygon(stacked[0], stacked[1], self._shape)

    def update_delta(self, delta_center, delta_theta):
        """
        Update using relative movement of center and theta
        Turn first and then Move next
        """
        self._last_center = self._center
        self._last_theta = self._theta
        new_theta = (self._theta + delta_theta) % (2*np.pi)
        rot_ma = tools.rotation_matrix(new_theta)
        speed_vec = rot_ma.dot(delta_center)
        new_center = self._center + speed_vec
        self.update_pos(new_center, new_theta)

    def hit_wall(self):
        """
        Call only once after update_delta
        Will change back to last center/theta 
        and add hit_wall punishment to reward
        """
        self.update_pos(self._last_center, self._last_theta)
        self._reward += self._reward_dict['hit_wall']

    @property
    def eye(self):
        """(left_eye_pos, right_eye_pos, theta, beta)"""
        return (self._lt_f.copy(), self._rt_f.copy(), self._theta, self._beta)

    @property
    def reward(self):
        """
        return collected rewards
        """
        return self._reward

    @property
    def pos(self):
        """
        return center position
        """
        return self._center.copy()

    @property
    def nose(self):
        """
        return nose position
        """
        return self._nose_pos.copy()

    def ate_apple(self):
        return self._ate_apple

    def is_dead(self):
        return self._dead

    def reset_reward(self):
        """
        Reset reward to 0
        Call this after 
        """
        self._reward = 0
        self._ate_apple = False

    def collided(self, t_type):
        if t_type == tt.Apple:
            self._reward += self._reward_dict['eat_apple']
            self._ate_apple = True