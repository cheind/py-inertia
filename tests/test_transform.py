import numpy as np

import inertia
import numpy.linalg as lin

from numpy.testing import assert_array_almost_equal as aae

def test_identity():
    aae(inertia.identity(2), np.eye(3))
    aae(inertia.identity(3), np.eye(4))

def test_decompose():
    m = np.asarray([[1,2,3], [4,5,6], [7,8,9]])
    r,tr = inertia.decompose(m)
    aae(r, [[1,2],[4,5]])
    aae(tr, [3,6])

def test_rotate_ccw():
    m = inertia.rotate(angle=np.pi / 2)
    aae(m, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])

def test_rotate_cw():
    m = inertia.rotate(angle=-np.pi / 2)
    aae(m, [[0, 1, 0],[-1, 0, 0], [0, 0, 1]])

def test_translate():
    m = inertia.translate(offset=[-10, 10])
    aae(m, [[1, 0, -10],[0, 1, 10], [0, 0, 1]])

def test_inverse():
    m = inertia.translate(offset=[1, 1]) @ inertia.rotate(angle=np.pi / 2)
    aae(inertia.inv(m), lin.inv(m))
    aae(inertia.inv(m, hint='isometry'), lin.inv(m))

def test_h():
    aae(inertia.h([1,1]), [1,1,1])
    aae(inertia.h([1,1], s=0), [1,1,0])

    aae(inertia.h([[1,1],[1,1]]), [[1, 1],[1, 1],[1, 1]])
    aae(inertia.h([[1,1],[1,1]], s=0), [[1, 1],[1, 1],[0, 0]])
    aae(inertia.h([[1,1],[1,1]], axis=1, s=0), [[1, 1, 0],[1, 1, 0]])

def test_hn():
    aae(inertia.hn([1,1,1]), [1,1])
    aae(inertia.hn([1,1]), [1])
    aae(inertia.hn([[1,1],[1,1],[0,0]]), [[1,1], [1,1]])
    aae(inertia.hn([1,1,2]), [0.5, 0.5])
    
def test_transform_points():    
    m = inertia.translate(offset=[-10, -10]) @ inertia.rotate(angle=np.pi/2)
    aae(inertia.tp(m, [1, 1]), [-11, -9])
    aae(inertia.tp(m, [[1, 2],[1, 2]]), [[-11,-12],[-9,-8]])

def test_transform_vectors():    
    m = inertia.translate(offset=[-10, -10]) @ inertia.rotate(angle=np.pi/2)   
    aae(inertia.tv(m, [1, 1]), [-1, 1])
    aae(inertia.tv(m, [[1, 2],[1, 2]]), [[-1,-2],[1,2]])
