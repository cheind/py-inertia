
from inertia import soa
import math

class Items(object):
    def __init__(self):
        self.__dict__['_all'] = []

    def __setattr__(self, name, value):
        self._all.append(value)
        super(Items, self).__setattr__(name, value)

    def __iter__(self):
        return iter(self._all)
        
class World2d(object):

    BodySOA = soa.create('BodySOA', [
        soa.Field('inverse_mass', shape=(1,), value=1),
        soa.Field('inverse_inertia', shape=(1,), value=1),
        soa.Field('position', shape=(2,)),
        soa.Field('velocity', shape=(2,)),
        soa.Field('acceleration', shape=(2,)),
        soa.Field('orientation', shape=(1,)),
        soa.Field('angular_velocity', shape=(1,)),
        soa.Field('angular_acceleration', shape=(1,))
    ])

    def __init__(self, body_capacity=10):
        self.bodies = Items()
        self.forces = Items()        
        self.body_soa = World2d.BodySOA(body_capacity)    

        self.time = 0.
        """Current simulation time."""
    
    def update(self, timestep):
        """Updates simulation by advancing `timestep` seconds in one sweep."""

        # Clear accumulators
        self.body_soa.acceleration.fill(0.)
        self.body_soa.angular_acceleration.fill(0.)

        # Apply forces
        for f in self.forces:
            if f.active:
                f.apply(self, timestep)

        # Euler integration
        self.body_soa.velocity += self.body_soa.acceleration * timestep
        self.body_soa.angular_velocity += self.body_soa.angular_acceleration * timestep

        self.body_soa.position += self.body_soa.velocity * timestep
        self.body_soa.orientation += self.body_soa.angular_velocity * timestep

        self.time += timestep

    def run_for(self, duration, timestep):
        n = math.floor(duration / timestep)
        r = duration - n * timestep

        for _ in range(n):
            self.update(timestep)
        self.update(r)

    
