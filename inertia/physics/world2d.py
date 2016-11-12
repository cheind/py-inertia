
from inertia import soa
import math

class Bodies:
    def __init__(self):
        pass

    def __setattr__(self, name, value):
        if isinstance(value, Body2d):
            super(Bodies, self).__setattr__(name, value)
        else:
            raise Exception('Cannot add {}. Not a Body2d'.format(name))
        
class World2d(object):

    BodySOA = soa.create('BodySOA', [
        soa.Field('inverse_mass', shape=(1,), value=1),
        soa.Field('inverse_inertia', shape=(1,), value=1),
        soa.Field('position', shape=(2,)),
        soa.Field('velocity', shape=(2,)),
        soa.Field('acceleration', shape=(2,)),
        soa.Field('orientation', shape=(1,)),
        soa.Field('angular_velocity', shape=(1,)),
        soa.Field('angular_acceleration', shape=(1,)),
        soa.Field('linear_force_accumulator', shape=(2,)),
        soa.Field('torque_accumulator', shape=(1,))
    ])

    def __init__(self, body_capacity=10):
        self.bodies = Bodies()
        self.body_soa = World2d.BodySOA(body_capacity)

    def update(self, timestep):
        """Updates all entities by timestep."""
                
        # Update linear acceleration by a = F / m, where F is the net force.
        self.body_soa.acceleration += self.body_soa.linear_force_accumulator * self.body_soa.inverse_mass

        # Update angular acceleration by a = inv(I) * T, where T is the net torque.
        self.body_soa.angular_acceleration += self.body_soa.torque_accumulator * self.body_soa.inverse_inertia 

        # Euler integration
        self.body_soa.velocity += self.body_soa.acceleration * timestep
        self.body_soa.angular_velocity += self.body_soa.angular_acceleration * timestep

        self.body_soa.position += self.body_soa.velocity * timestep
        self.body_soa.orientation += self.body_soa.angular_velocity * timestep

        # Clear accumulators
        self.body_soa.linear_force_accumulator.fill(0.)
        self.body_soa.torque_accumulator.fill(0.)

    def run_for(self, duration, timestep):
        n = math.floor(duration / timestep)
        r = duration - n * timestep

        for _ in range(n):
            self.update(timestep)
        self.update(r)
    