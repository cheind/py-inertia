
import math
import numpy as np

from inertia import transform as t
from inertia.physics.world2d import World2d

class Body2d(World2d.BodySOA.View):
    """A rigid body"""

    def __init__(self, world, id=None, mass=1, inertia=1):
        if id is None:
            id = world.body_soa.take()
            new = True

        super(Body2d, self).__init__(world.body_soa, id)

        if new:
            self.inverse_mass = 1.0 / mass
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

    
        
