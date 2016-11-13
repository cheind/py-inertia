import numpy as np


from inertia.physics.world2d import World2d
from inertia.physics.body2d import Body2d
from inertia.physics.force2d import ConstantForce
import inertia as i

from numpy.testing import assert_array_almost_equal as aae


def pos(p0, v0, a, t):
    """Returns the position from constant acceleration"""
    return np.asarray(p0) + t*np.asarray(v0) + 0.5 * np.asarray(a) * t * t

def test_newton_one():
    # In an inertial reference frame, an object either remains at rest 
    # or continues to move at a constant velocity, unless acted upon by a net force

    w = World2d(1)
    b = Body2d(w, mass=1)

    w.forces.x = ConstantForce(w, b, [1, 0], tbegin=0.0, tend=1.0)

    # At rest
    aae(b.position, [0, 0])    
    w.run_for(1, 0.0001)

    print(b.position)
    print(w.time)
    print(pos([0,0], [0,0], [1,0], 1))

    aae(b.position, [0, 0])






    """
    # One time net force
    b.add_force([1, 0]) # One time force
    w.run_for(0.5, 0.001)
    print(b.position)
    aae(b.position, pos([0,0], [0,0], [1,0], 0.5), decimal=3)

    b.add_force([-2, 0]) # One time force
    pred = pos(b.position, b.velocity, [-1, 0], 0.5)  
    w.run_for(0.5, 0.001)
    aae(b.position, pred, decimal=3)
    """