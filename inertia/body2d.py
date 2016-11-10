
import math
import numpy as np

from . import transform as t
from . import graph as g


class Body2d(g.Node):
    """A rigid body"""

    def __init__(self, mass=1, inertia=1, pose=t.identity(2)):
        super(Body2d, self).__init__(pose)

        self.invMass = 1.0 / mass
        """The body's inverse mass"""
        
        self.invInertia = 1.0 / inertia
        """The body's inverse inertia coefficient"""
       
        self.velocity = np.zeros(2)
        """The body's linear velocity"""

        self.acceleration = np.zeros(2)
        """The body's linear acceleration"""

        self.angular_velocity = 0.
        """The body's angular velocity"""

        self.angular_acceleration = 0.
        """The body's angular acceleration"""

        self._facc = np.zeros((1, 2))
        """The body's linear force accumulator"""

        self._tacc = np.zeros((1, 1))
        """The body's torque accumulator"""
       
               
    @property
    def movable(self):
        return self.invMass != 0
    
    @property
    def mass(self):
        return float('inf') if not self.movable else 1.0 / self.invMass

    @property
    def inertia(self):
        return 1.0 / self.invInertia

    def update(self, tdelta):
        """Integrates the rigid body forward in time by the given amount."""
                
        # Update linear acceleration by a = F / m, where F is the net force.
        self.acceleration += self._facc * self.invMass

        # Update angular acceleration by a = inv(I) * T, where T is the net torque.
        self.angular_acceleration += self.invInertia * self._tacc

        # Euler integration
        self.velocity += self.acceleration * tdelta
        self.angular_velocity += self.angular_acceleration * tdelta

        p = self.velocity * tdelta
        o = self.angular_velocity * tdelta

        # Update pose
        self.pose = self.pose @ t.translate(offset=p) @ t.rotate(angle=o)

        # Clear accumulators
        self._facc.fill(0.)
        self._tacc.fill(0.)

    def add_force(self, force, point=None):
        """Adds a new force to the bodies accumulator.

        Both the force vector and an optional point are assumed to be 
        in  
        """

        # Force affecting linear motion
        self._facc += force

        # Force affecting angular motion
        if point is not None:
            self._tacc += np.cross(point - self.position, force)
        
