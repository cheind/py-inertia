
import numpy as np
from . import transform as t

class Node:

    def __init__(self, pose=t.identity(dims=2), parent=None, name=None):
        self.pose = pose
        self.parent = parent
        self.name = name
        self.children = {}

    def __getitem__(self, key):
        return self._create(key)
    
    def __setitem__(self, key, value):
        self._create(key, value=value)
                
    def _create(self, key, value=None):
        parts = key.split('.')
        elems = parts if not value else parts[:-1]       
        
        n = self
        for p in elems:
            if not p in n.children:
                n.children[p] = Node(parent=n, pose=t.identity(dims=self.pose.shape[0]))
            n = n.children[p]

        if value:
            value.parent = n
            n.children[parts[-1]] = value
            n = value

        return n
        
    @property
    def pose_in_world(self):
        m = t.identity(dims=self.pose.shape[0] - 1)
        node = self
        while node is not None:
            m = np.dot(node.pose, m)
            node = node.parent
        return m

    def pose_in(self, other):
        t_wn = self.pose_in_world
        t_wo = other.pose_in_world
        return np.dot(t.inv(t_wo), t_wn)
