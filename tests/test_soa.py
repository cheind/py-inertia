import numpy as np

from inertia import soa
from numpy.testing import assert_array_almost_equal as aae



fields = [
    soa.Field('x', dtype=np.float64, shape=(1,3)),
    soa.Field('y', dtype=np.float64, shape=(3,)),
    soa.Field('z', dtype=np.int, shape=(3,1)),
    soa.Field('w', dtype=np.float64, shape=(3,3)),
    soa.Field('active', dtype=bool),
]

SOA = soa.create('SOA', fields)

def test_soa_shape():
    mysoa = SOA(2)

    assert mysoa.x.shape == (2, 3)
    assert mysoa.x.dtype == np.float64
    assert np.all(mysoa.x == 0)

    assert mysoa.y.shape == (2, 3)
    assert mysoa.y.dtype == np.float64
    assert np.all(mysoa.y == 0)

    assert mysoa.z.shape == (3, 2)
    assert mysoa.z.dtype == np.int
    assert np.all(mysoa.z == 0)

    assert mysoa.w.shape == (2, 3, 3)
    assert mysoa.w.dtype == np.float64
    assert np.all(mysoa.w == 0)

    assert mysoa.active.shape == (2, )
    assert mysoa.active.dtype == bool
    assert not np.any(mysoa.active)

def test_soa_resize():
    mysoa = SOA(0)

    assert len(mysoa) == 0
    assert mysoa.capacity == 0
     
    mysoa.take()

    assert len(mysoa) == 1
    assert mysoa.capacity == 2

    mysoa.take()

    assert len(mysoa) == 2
    assert mysoa.capacity == 2

    mysoa.x.fill(1.)
    mysoa.active.fill(True)

    print(mysoa.x)

    mysoa.take()

    print(mysoa.x)

    assert len(mysoa) == 3
    assert mysoa.capacity == 6

    

    assert np.all(mysoa.x[:2, :] == 1.)



class DerivedView(SOA.View):
    def __init__(self, soa, id=None):
        super(DerivedView, self).__init__(soa, soa.take() if id is None else id)

    @property
    def x_plus_two(self):
        return self.x + 2
