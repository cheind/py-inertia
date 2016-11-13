
from enum import Enum
import math
import numpy as np

from inertia import view
from inertia import transform as t

class Body2dFields(Enum):
    position = ('position', float, 2)
    linear_velocity = ('linear_velocity', float, 2)
    linear_acceleration = ('linear_acceleration', float, 2)
    orientation = ('orientation', float, 1)
    angular_velocity = ('angular_velocity', float, 1)
    angular_acceleration = ('angular_acceleration', float, 1)
    inverse_mass = ('inverse_mass', float, 1)
    inverse_inertia = ('inverse_inertia', float, 1)

Body2dView = view.create_view('Body2dView', Body2dFields)

class Body2d(Body2dView):
    """One or more rigid bodies"""

    def __init__(self, world, id, mass=None, inertia=None):        
        self.id = id
        super(Body2d, self).__init__(world.bodies, self.id)
        
        if mass is not None:
            self.inverse_mass = 1.0 / mass
        if inertia is not None:
            self.inverse_inertia = 1.0 / inertia
        
    @property
    def mass(self):
        return float('inf') if self.inverse_mass == 0 else 1.0 / self.inverse_mass

    @property
    def inertia(self):
        return float('inf') if self.inverse_inertia == 0 else 1.0 / self.inverse_inertia

    @property
    def pose(self):
        return t.translate(offset=self.position) @ t.rotate(angle=self.orientation)
    
        
