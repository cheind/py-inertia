from enum import Enum
import types

import numpy as np

class FieldType(Enum):
    """Enumeration of supported numpy field types in SOA."""

    scalar = 1
    array = 2
    row_vector = 4
    col_vector = 8
    matrix = 16
    tensor = 32

    @staticmethod
    def classify(s):
        """Returns the field type from the given shape tuple."""
        if len(s) == 1: 
            return FieldType.scalar if s[0] == 1 else FieldType.array
        elif len(s) == 2:
            if s[0] == 1 and s[1] > 1:
                return FieldType.row_vector
            elif s[0] > 1 and s[1] == 1:
                return FieldType.col_vector
            else:
                return FieldType.matrix
        elif len(s) > 2:
            return FieldType.tensor
        else:
            raise Exception("Unknown shape")

def default_view_property(self):
    def col_vector():
        def getter(view):
            return getattr(view.soa, self.name)[:, view.id]
        
        def setter(view, value):
            getattr(view.soa, self.name)[:, view.id] = value

        return property(getter, setter)

    def default():
        def getter(view):
            return getattr(view.soa, self.name)[view.id]
        
        def setter(view, value):
            getattr(view.soa, self.name)[view.id] = value

        return property(getter, setter)

    return col_vector() if self.stype == FieldType.col_vector else default() 

class Field(object):
    """Describes one numerical entry of a SOA."""
    def __init__(self, name, shape=(1,), dtype=np.float64, value=0, view_property=default_view_property):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.fill_value = value
        self.stype = FieldType.classify(shape)
        self.compute_shape = self.shape_fnc(self.stype)
        self.view_property = types.MethodType(view_property, self) # Bind method
        
    def shape_fnc(self, stype):
        """Returns a function for computing a field's shape for a given capacity."""
        return {
            FieldType.scalar: lambda n, s: (n,),
            FieldType.array: lambda n, s: (n, s[0]),
            FieldType.row_vector: lambda n, s: (n, s[1]),
            FieldType.col_vector: lambda n, s: (s[0], n),
            FieldType.matrix: lambda n, s: (n,) + s,
            FieldType.tensor: lambda n, s: (n,) + s
        }[stype]

    def create_numpy(self, n):
        """Returns a numpy array for given field that holds at least `n` elements."""
        return np.full(self.compute_shape(n, self.shape), self.fill_value, dtype=self.dtype)

    def resize_numpy(self, a, n):
        """Resizes a numpy field array `a` to hold at least `n` elements."""
        old_size = a.shape
        new_size = self.compute_shape(n, self.shape)
        diff = np.subtract(new_size, old_size)

        # Need to create a tuple (pad-before, pad-after) for each axis
        return np.pad(a, [(0,d) for d in diff], mode='constant', constant_values=self.fill_value)

class SOABase(object):
    """Base class for structure of array (SOA) types."""
    
    def __init__(self, fields, capacity=0):
        """Initialize from fields and initial capacity."""
        self.fields = fields
        self.n = 0
        self.capacity = capacity
        for f in fields:
            setattr(self, f.name, f.create_numpy(capacity))


    def __len__(self):
        """Returns the number of taken items."""
        return self.n
        
    def take(self):
        """Returns the id of the next free slot in this SOA."""
        if self.n == self.capacity:
            new_capacity = (self.capacity + 1) * 2
            for f in self.fields:
                new_f = f.resize_numpy(getattr(self, f.name), new_capacity)
                setattr(self, f.name, new_f)
            self.capacity = new_capacity

        id = self.n
        self.n += 1
        return id

    def view(self, id):
        """Returns a structured view for the slot at `id`.
        
        Each generated SOA class has a default view class associated with
        it. If the SOA class is `X`, then its default view class is named
        `XView` and is accessible via `X.View`.

        The default view class provides a structured reference to all
        of the objects properties. The properties do not hold copies
        of the underlying SOA values but reference them. Copies are
        avoided to avoid invalidated references once the underlying SOA
        resizes arrays.
        """
        return self.__class__.View(self, id)

    def view_as(self, id, klass, *args, **kwargs):
        """Returns a custom structured view for tje slot at `id`."""
        return klass(self, id, *args, **kwargs)

class SOAViewBase(object):
    """The base class for views on SOA objects.

    While the SOA class maintains object properties distributed
    over multiple arrays, a view aggregates these properties into
    a structured representation. A view never actually stores values,
    put merely interacts with the underlying SOA.
    """

    def __init__(self, soa, id):
        self.soa = soa
        self.id = id

def create_view(cls_name, fields):
    """Returns a structured view class for the given SOA fields."""

    the_view = {}
    
    def init(self, soa, id):
        SOAViewBase.__init__(self, soa, id)

    the_view['__init__'] = init
    
    for f in fields:
        the_view[f.name] = f.view_property()

    view_cls = type(cls_name, (SOAViewBase,), the_view)
    return view_cls


def create(cls_name, fields):
    """Returns a new SOA class generated from the given fields."""

    def soa_init(self, capacity=0):
        SOABase.__init__(self, fields, capacity)

    soa_cls = type(cls_name, (SOABase,), {"__init__": soa_init})
    soa_cls.View = create_view(cls_name + 'View', fields)

    return soa_cls
