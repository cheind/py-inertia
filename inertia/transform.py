
import numpy as np
import numpy.linalg as lin

def identity(dims=2):
    return np.eye(dims+1, dims+1)

def decompose(pose):
    return pose[:-1, :-1], pose[:-1, -1]

def rotate(angle=None):   
    if np.isscalar(angle):
        # two-dimensional case        
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise Exception("Not yet implemented")

def translate(offset=None):
    if isinstance(offset, (list, tuple, np.ndarray)):
        dims = len(offset)
        tr = identity(dims)
        tr[:dims, dims] = offset
        return tr
    else:
        raise Exception("Not yet implemented")

def inv(pose, hint=None):
    if hint == 'isometry':
        r, t = decompose(pose)
        ir = np.transpose(r)
        it = np.dot(-ir, t)
        
        f = identity(dims=r.shape[0])        
        f[:-1, :-1] = ir
        f[:-1, -1] = it
        return f
    elif hint is None:
        return lin.inv(pose)
    else:
        raise Exception("Unknown hint")

def h(x, axis=0, s=1):
    """Convert to homogeneous coordinates."""
    return np.apply_along_axis(lambda a: np.concatenate((a, [s])), axis, x)

def hn(x, axis=0):
    """Normalize homogeneous coordinates."""

    def _hn(a):
        l = a[-1]
        s = 1.0 / l if l != 0 else 1.0
        return a[:-1] * s

    return np.apply_along_axis(_hn, axis, x)

def tp(pose, x):
    return hn(np.dot(pose, h(x)))

def tv(pose, x):
    return hn(np.dot(pose, h(x, s=0)))