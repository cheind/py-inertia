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
    mysoa = SOA(1)

    assert len(mysoa) == 0
    assert mysoa.capacity == 1
     
    mysoa.take()

    assert len(mysoa) == 1
    assert mysoa.capacity == 1

    assert mysoa.x.shape == (1, 3)
    assert mysoa.y.shape == (1, 3)
    assert mysoa.z.shape == (3, 1)
    assert mysoa.w.shape == (1, 3, 3)
    assert mysoa.active.shape == (1, )

    mysoa.x.fill(1.)
    mysoa.z.fill(1.)
    mysoa.active.fill(True)

    mysoa.take()

    assert len(mysoa) == 2
    assert mysoa.capacity == 4

    assert mysoa.x.shape == (4, 3)
    assert mysoa.y.shape == (4, 3)
    assert mysoa.z.shape == (3, 4)
    assert mysoa.w.shape == (4, 3, 3)
    assert mysoa.active.shape == (4, )

    assert np.all(mysoa.x[:1, :] == 1)
    assert np.all(mysoa.x[1:, :] == 0)
    assert np.all(mysoa.z[:, :1] == 1)
    assert np.all(mysoa.z[:, 1:] == 0)
    assert np.all(mysoa.active[:1])
    assert not np.any(mysoa.active[1:])

def test_default_view():
    mysoa = SOA(0)

    v0 = mysoa.view(mysoa.take())
    v0.x = 2
    v0.z = [1,2,3]

    v1 = mysoa.view(mysoa.take())
    v1.active = False
    v1.x = [1,0,0]
    v1.w = np.eye(3)

    v2 = mysoa.view(mysoa.take())
    v2.x = [4,4,4]
    v2.w = np.full((3,3), 10, dtype=float)

    aae(v0.x, [2,2,2])
    aae(v0.z, [1,2,3])
    aae(v1.active, False)
    aae(v1.w, np.eye(3))
    aae(v2.x, [4,4,4])
    aae(v2.w, 10)

def test_views_reflect_changes():
    mysoa = SOA(1)

    v0 = mysoa.view(mysoa.take())
    v0.x = 2

    v1 = mysoa.view(v0.id)
    aae(v1.x, 2)
    v1.x = 3
    aae(v0.x, 3)


class DerivedView(SOA.View):
    def __init__(self, soa, id=None):
        super(DerivedView, self).__init__(soa, soa.take() if id is None else id)

    @property
    def x_plus_two(self):
        return self.x + 2

def test_derived_views():
    mysoa = SOA(1)

    d0 = DerivedView(mysoa)
    d0.x = 2
    aae(d0.x_plus_two, 4)

    d1 = mysoa.view_as(0, DerivedView)
    d1.x = 3
    aae(d0.x_plus_two, 5)
