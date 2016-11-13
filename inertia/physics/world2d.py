import math
import numpy as np

from inertia.physics.body2d import Body2d, Body2dFields
from inertia import transform as t

class Items(object):
    def __init__(self):
        self.__dict__['_all'] = []

    def __setattr__(self, name, value):
        self._all.append(value)
        super(Items, self).__setattr__(name, value)

    def __iter__(self):
        return iter(self._all)
        
class World2d(object):

    def __init__(self, body_capacity=10):
        self.bodies = np.zeros(body_capacity, [e.value for e in Body2dFields])
        self.initialize_bodies(np.s_[:body_capacity])
        self.forces = Items()        

        self.body_count = 0
        """The number of bodies."""

        self.body_capacity = body_capacity
        """Body capacity without resizing."""

        self.time = 0.
        """Current simulation time."""

    def new_body(self, **kwargs):
        if self.body_count == self.body_capacity:
            new_capacity = (self.body_capacity + 1) * 2
            self.bodies.resize(new_capacity, refcheck=False)
            self.initialize_bodies(np.s_[-(new_capacity - self.body_capacity)])
            self.body_capacity = new_capacity

        b = Body2d(self, id=self.body_count, **kwargs)
        self.body_count += 1
        return b

    def initialize_bodies(self, slice):
        b = Body2d(self, slice)
        b.inverse_mass = 1.
        b.inverse_inertia = 1.
    
    def update(self, timestep):
        """Updates simulation by advancing `timestep` seconds in one sweep."""

        b = self.bodies[:self.body_count].view(np.recarray)

        # Clear accumulators
        b.linear_acceleration.fill(0.)
        b.angular_acceleration.fill(0.)

        # Apply forces
        for f in self.forces:
            if f.active:
                f.apply(self, timestep)

        # Euler integration
        b.linear_velocity += b.linear_acceleration * timestep
        b.angular_velocity += b.angular_acceleration * timestep

        b.position = b.linear_velocity * timestep
        b.orientation = b.angular_velocity * timestep

        self.time += timestep

    def run_for(self, duration, timestep):
        n = math.floor(duration / timestep)
        r = duration - n * timestep

        for _ in range(n):
            self.update(timestep)
        self.update(r)

        

    
