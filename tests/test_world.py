import numpy as np


from inertia.physics.world2d import World2d
from inertia.physics.body2d import Body2dFields, Body2d
from inertia.physics.force2d import Force2d, ConstantForce
from inertia import view

from numpy.testing import assert_array_almost_equal as aae


def test_body_construction():
    w = World2d(20)
    for _ in range(10000):
        b = w.new_body(mass=1)
    print(w.body_count)
    #w.forces.x = ConstantForce([1,0], w, bodies=b)
    w.run_for(10, 0.001)

    print(b.position)