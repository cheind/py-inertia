
import math
import numpy as np

from . import pose2d
from .pose2d import Frame


class Body2d:
    """A rigid body"""

    def __init__(self, mass=1, inertia=1, pose=pose2d.identity()):
        self.invMass = 1.0 / mass
        """The body's inverse mass"""
        
        self.invInertia = 1.0 / inertia
        """The body's inverse inertia coefficient"""

        self.pose = np.copy(pose)
        """The body's current pose"""
        
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        """The body's position, linear velocity and acceleration"""

        self.orientation = 0.
        self.angular_velocity = 0.
        self.angular_acceleration = 0.
        """The body's orientation, angular velocity and acceleration"""

        self._facc = np.zeros((1, 2))
        """The body's linear force accumulator"""

        self._tacc = np.zeros((1, 1))
        """The body's torque accumulator"""

        self.pose = pose2d.identity()
        """The body's current pose"""

        r,t = pose2d.decompose(pose)
        self.position = np.copy(t)
        self.orientation = math.atan2(r[1,0], r[0, 1])
               
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

        self.position += self.velocity * tdelta
        self.orientation += self.angular_velocity * tdelta

        # Update pose
        self.pose = pose2d.chain(pose2d.t(offset=self.position), pose2d.r(angle=self.orientation))

        # Clear accumulators
        self._facc.fill(0.)
        self._tacc.fill(0.)

    def add_force(self, force, point=None, frame_force='world', frame_point='body'):
        """Adds a new force to the bodies accumulator"""

        # Force affecting linear motion
        if Frame(frame_force) == Frame.body:
            force = pose2d.transform_vector(self.pose, force)

        self._facc += force

        # Force affecting angular motion
        if point is None:
            return
        
        if Frame(frame_point) == Frame.body:
            point = pose2d.transform_point(self.pose, point)

        self._tacc += np.cross(point - self.position, force)
        
