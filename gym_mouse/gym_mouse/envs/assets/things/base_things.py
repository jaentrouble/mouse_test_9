import numpy as np

class Base():
    """
    Base object

    Every object has its own indices and color.
    Indices are from skimage.draw, as (rr, cc) form.
    Indices are in 'indices' property and returns (rr, cc)
    It is interpretable as img[rr,cc,:] = color

    It doesn't care about the screen size by default.
    """
    def __init__(self):
        self._rr = None
        self._cc = None
        # For efficiency of drawing things, save last indices if changed
        self._last_rr = None
        self._last_cc = None
        # Color can be a tuple or an array of color per pixel
        self._color = None
        self._is_updated = False
        # Type of this thing
        self._t_type = None

    @property
    def t_type(self):
        return self._t_type

    @property
    def indices(self):
        """Get current indices"""
        if self._rr is None or self._cc is None:
            raise NotImplementedError('No rr or cc')
        return self._rr, self._cc
    
    @indices.setter
    def indices(self, indices):
        """
        Gets indices and saves last indices
        indices : (rr, cc)
        """
        self._last_rr, self._last_cc = self._rr, self._cc
        self._is_updated = True
        self._rr, self._cc = indices

    @property
    def last_indices(self):
        return self._last_rr, self._last_cc

    @property
    def color(self):
        """Get current color"""
        if self._color is None:
            raise NotImplementedError('No color')
        return self._color

    @color.setter
    def color(self, color):
        """
        Set color
        color : (R, G, B)
        This will be changed to np.uint8, so be careful for overflow
        """
        self._color = np.array(color, dtype=np.uint8)

    @property
    def is_updated(self):
        """
        Returns True if the indices have changed
        """
        return self._is_updated

    def reset_updated(self):
        """
        Set is_updated to False
        """
        self._is_updated = False

    def hit_wall(self):
        """
        Override this method if needed.
        """
        pass

    def update(self):
        """
        Override this method if needed.
        This will be called every time.
        Dynamic objects should use this method to move themselves.
        """
        pass

    def collided(self, t_type):
        """
        Define actions to do when collided with 'type' thing
        This may be called TWICE if two dynamic objects collide
        """
        pass