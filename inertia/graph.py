
import numpy as np
from . import pose2d

class PoseNode:

    def __init__(self, pose=pose2d.identity(), parent=None, name=None):
        self._pose = pose
        self._name = name
        self._children = []
        self._parent = parent
        
    def __getattr__(self, name):
        n = PoseNode(parent=self, name=name)
        self._children.append(n)
        setattr(self, name, n)
        return n

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        self._pose = value
   
    @property
    def wrt_world(self):
        m = pose2d.identity()
        n = self
        while n is not None:
            m = np.dot(n._pose, m)
            n = n._parent
        return m
    
    def wrt(self, frame):
        t_ws = self.wrt_world
        t_wo = frame.wrt_world
        return np.dot(pose2d.inv(t_wo), t_ws)


from enum import Enum


class WrtType(Enum):
    points = 'points'
    vectors = 'vectors'

class WrtView(np.ndarray):

    def wrt(self, frame):
        pose = self._node.wrt(frame)        
        if self._type == WrtType.points:
            t = pose2d.transform_points(pose, self)
            return ref(t, frame, self._type)
        else:
            t = pose2d.transform_vectors(pose, self)
            return ref(t, frame, self._type)

def ref(x, frame, as_type='points'):
    w = np.asarray(x).view(type=WrtView)
    w._type = WrtType(as_type)
    w._node = frame
    return w

