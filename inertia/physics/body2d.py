
import math
import numpy as np

from inertia import transform as t
from inertia.physics.world2d import World2d

class Body2d(World2d.BodySOA.View):
    """A rigid body"""

    def __init__(self, world, id=None, mass=1, inertia=1):
        super(Body2d, self).__init__(world.body_soa, id)

        if id is None:
            # Initialize body
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

    def add_force(self, force, point=None):
        """Adds a new world force to the bodies accumulator.
        
        A force applied by this method affects the next simulation timestep only.
        Depending on the duration of the next update, this force might have a bigger
        or smaller effect on the body.
        """

        # Force affecting linear motion
        self.linear_force_accumulator += force

        # Force affecting angular motion
        if point is not None:
            self.torque_accumulator += np.cross(point - self.position, force)

    
        
