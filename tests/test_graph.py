import numpy as np
from numpy.testing import assert_array_almost_equal as aae

import inertia

def test_construction():
    g = inertia.graph.Node(name='foo')
    aae(g.pose, inertia.identity(dims=2))
    assert g.name == 'foo'
    assert g.parent == None

def test_lazy_node_creation():
    g = inertia.graph.Node()
    g['x.y'].name ='foo'    
    assert isinstance(g['x'], inertia.graph.Node)
    assert isinstance(g['x']['y'], inertia.graph.Node)
    assert g['x']['y'].name == 'foo'
    
class X(inertia.graph.Node):    
    def __init__(self, value):
        self.value = value
        super(X, self).__init__()

def test_custom_node():
    g = inertia.graph.Node()
    g['x.y'] = X(3)
    assert isinstance(g['x'], inertia.graph.Node)
    assert isinstance(g['x']['y'], X)
    assert g['x.y'].value == 3
   
    g['x.y.z'] = X(4)
    assert isinstance(g['x'], inertia.graph.Node)
    assert isinstance(g['x']['y'], X)
    assert isinstance(g['x']['y']['z'], X)
    assert g['x.y'].value == 3
    assert g['x.y.z'].value == 4

def test_pose_computation():
    g = inertia.graph.Node()

    g['x'].pose = inertia.translate(offset=[10,10])
    g['x.z'].pose = inertia.translate(offset=[10,10]) @ inertia.rotate(angle=np.pi/2)
    g['y'].pose = inertia.translate(offset=[-10,-10])

    aae(g.pose_in_world, inertia.identity(dims=2))
    aae(g['x'].pose_in_world, inertia.translate(offset=[10,10]))
    aae(g['x.z'].pose_in_world, inertia.translate(offset=[10,10]) @ inertia.translate(offset=[10,10]) @ inertia.rotate(angle=np.pi/2))
    aae(g['x.z'].pose_in(g['y']), inertia.inv(g['y'].pose_in_world) @ inertia.translate(offset=[10,10]) @ inertia.translate(offset=[10,10]) @ inertia.rotate(angle=np.pi/2))
    aae(g.pose_in(g['x.z']), inertia.inv(g['x.z'].pose_in_world))