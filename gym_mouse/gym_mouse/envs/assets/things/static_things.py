from .base_things import Base
from ...constants import colors, rng
from .things_consts import DefaultSize as ds
from .things_consts import ThingsType as tt
from skimage import draw
import numpy as np

class Apple(Base):
    """
    Apple
    Red Circle shaped object
    Will be generated at a random place when eaten
    """
    def __init__(self, center, shape, radius = ds.Apple_radius):
        """
        center: Center coordiate of the apple
        radius: Radius of the apple (Default = 10)
        shape : Maximum size of the grid
        """
        super().__init__()
        self._shape = shape
        self.reset(center, radius)
        self.color = colors.COLOR_APPLE
        self._t_type = tt.Apple

    @property
    def pos(self):
        """
        Center position
        """
        return self._center.copy()

    @pos.setter
    def pos(self, center):
        self._center = np.array(center)
    
    def collided(self, t_type):
        if t_type == tt.Mouse:
            self.is_eaten = True

    def reset(self, center, radius = ds.Apple_radius):
        self.pos = center
        self.indices = draw.disk(center, radius, shape=self._shape)
        self.is_eaten = False

    def update(self):
        if self.is_eaten:
            self.reset(
                (rng.np_random.randint(0, self._shape[0]),
                 rng.np_random.randint(0, self._shape[1])),
            )