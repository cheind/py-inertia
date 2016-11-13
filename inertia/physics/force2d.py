import math
import numpy as np
import collections
from enum import Enum

from inertia import transform as t
from inertia.physics.world2d import World2d
from inertia.physics.body2d import Body2d


class ForceMode(Enum):
    force = 1
    """Affected by mass."""

    acceleration = 2
    """Not affected by mass."""

class Force2d(object):
    """Base class for forces."""

    def __init__(self, world, bodies, tbegin=0., tend=float('inf')):
        self.update_bodies(world, bodies)
        self.tbegin = tbegin
        self.tend = tend
        self.world = world

    def update_bodies(self, w, bodies):
        if bodies is None:
            self.bodies = None
        elif isinstance(bodies, collections.Iterable):
            self.bodies = w.soa_bodies.view([b.id for b in bodies])
        elif isinstance(bodies, int):
            self.bodies = w.soa_bodies.view(bodies)
        elif isinstance(bodies, tuple):
            self.bodies = w.soa_bodies.view(bodies)
        elif isinstance(bodies, Body2d):
            self.bodies = bodies
        elif isinstance(bodies, World2d.BodySOA):
            self.bodies = bodies
        else:
            raise Exception("Unknown body selection")

    @property
    def active(self):
        return self.tbegin <= self.world.time < self.tend and self.bodies is not None

    @staticmethod
    def apply_force(force, bodies, point=None, mode=ForceMode.force):
        if mode == ForceMode.force:
            bodies.acceleration += bodies.inverse_mass * force
        elif mode == ForceMode.acceleration:
            bodies.acceleration += force

        if point is not None:
            t = np.cross(point - bodies.position, force)
            Force2d.apply_torque(t, bodies, mode=mode)


    @staticmethod
    def apply_torque(torque, bodies, mode=ForceMode.force):
        if mode == ForceMode.force:
            bodies.angular_acceleration += bodies.inverse_inertia * torque
        elif mode == ForceMode.acceleration:
            bodies.angular_acceleration += torque

class ConstantForce(Force2d):
    def __init__(self, world, bodies, force, mode=ForceMode.force, point=None, tbegin=0., tend=float('inf')):
        super(ConstantForce, self).__init__(world, bodies, tbegin=tbegin, tend=tend)
        
        self.force = np.asarray(force)
        self.mode = mode
        self.point = point

    def apply(self, world, timestep):
        Force2d.apply_force(self.force, self.bodies, point=self.point, mode=self.mode)