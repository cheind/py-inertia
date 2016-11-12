
from .body2d import Body2d
from inertia import soa

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
        soa.Field('torque_accumulator', shape=(2,))
    ])

    def __init__(self, body_capacity=10):
        self.bodies = Bodies()
        self.body_soa = World2d.BodySOA(body_capacity)

    