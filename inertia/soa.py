from enum import Enum

import numpy as np

class FieldType(Enum):
    scalar = 1
    array = 2
    row_vector = 4
    col_vector = 8
    matrix = 16
    tensor = 32

    @staticmethod
    def classify(s):
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

class Field(object):
    def __init__(self, name, shape=(1,), dtype=np.float64, order='C'):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.order = order
        self.stype = FieldType.classify(shape)
        self.compute_shape = self.shape_fnc(self.stype)

    @staticmethod
    def shape_fnc(stype):
        return {
            FieldType.scalar: lambda n, s: (n,),
            FieldType.array: lambda n, s: (n, s[0]),
            FieldType.row_vector: lambda n, s: (n, s[1]),
            FieldType.col_vector: lambda n, s: (s[0], n),
            FieldType.matrix: lambda n, s: (n,) + s,
            FieldType.tensor: lambda n, s: (n,) + s
        }[stype]

    def create_numpy(self, n):
        return np.zeros(self.compute_shape(n, self.shape), dtype=self.dtype, order=self.order)

    def resize_numpy(self, a, n):
        a.resize(self.compute_shape(n, self.shape), refcheck=False)


class SOABase(object):
    
    def __init__(self, fields, capacity=0):
        self.fields = fields
        self.n = 0
        self.capacity = capacity
        for f in fields:
            setattr(self, f.name, f.create_numpy(capacity))
        
    def take(self):
        if self.n == self.capacity:
            new_capacity = self.capacity * 2
            for f in self.fields:
                f.resize_numpy(getattr(self, f.name), new_capacity)
            self.capacity = new_capacity

        id = self.n
        self.n += 1
        return id

    def view(self, id):
        return self.__class__.View(self, id)

class SOAViewBase(object):

    def __init__(self, soa, id):
        self.soa = soa
        self.id = id

def create_view(cls_name, fields):

    def col_vector_property(name, stype):
        def getter(self):
            return getattr(self.soa, name)[:, self.id]
        
        def setter(self, value):
            getattr(self.soa, name)[:, self.id] = value

        return property(getter, setter)

    def default_property(name, stype):
        def getter(self):
            return getattr(self.soa, name)[self.id]
        
        def setter(self, value):
            getattr(self.soa, name)[self.id] = value

        return property(getter, setter)

    prop_gens = {
        FieldType.col_vector: col_vector_property
    }

    the_view = {}
    
    def init(self, soa, id):
        SOAViewBase.__init__(self, soa, id)

    the_view['__init__'] = init
    
    for f in fields:
        the_view[f.name] = prop_gens.get(f.stype, default_property)(f.name, f.stype)

    view_cls = type(cls_name, (SOAViewBase,), the_view)
    return view_cls


def create(cls_name, fields):

    def soa_init(self, capacity=0):
        SOABase.__init__(self, fields, capacity)

    soa_cls = type(cls_name, (SOABase,), {"__init__": soa_init})
    soa_cls.View = create_view(cls_name + 'View', fields)

    return soa_cls

MySOA = create('MySOA', fields=[
    Field('pos', dtype=np.float64, shape=(2,)),
    Field('vel', dtype=np.float64, shape=(2,)),
    Field('x', dtype=np.float64, shape=(1,3), order='F'),
    Field('y', dtype=np.float64, shape=(3,1)),
    Field('z', dtype=np.float64, shape=(3,3)),
    Field('active', dtype=bool),
])

class Body(MySOA.View):
    def __init__(self, soa):
        super(Body, self).__init__(soa, soa.take())

    @property
    def modified_pos(self):
        return self.pos + 2


