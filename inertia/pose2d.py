
import numpy as np
from enum import Enum

class Frame(Enum):
    world = 'world'
    body = 'body'

def identity():
    return np.eye(3, 3)

def decompose(pose):
    return pose[:2, :2], pose[:2, 2]

def r(angle=None):   
    if angle is not None:        
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def t(offset=None):
    if offset is not None:
        return np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])

def inv(pose, rigid=True):
    if rigid:
        f = identity()
        r, t = decompose(pose)
        ir = np.transpose(r)
        it = np.dot(-ir, t)
        f[:2, :2] = ir
        f[:2, 2] = it
        return f
    else:
        return np.inv(pose)

def h(x, axis=0, value=1):
    """Convert to homogeneous coordinates."""
    return np.apply_along_axis(lambda a: np.concatenate((a, [value])), axis, x)

def hn(x, axis=0):
    """Normalize homogeneous coordinates."""
    return np.apply_along_axis(lambda a: a[:-1], axis, x)

def chain(*poses):
    f = identity()
    for p in poses:
        f = np.dot(f, p)
    return f

def transform_points(pose, x):
    return hn(np.dot(pose, h(x)))

def transform_vectors(pose, x):
    return hn(np.dot(pose, h(x, value=0)))