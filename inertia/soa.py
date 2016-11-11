import numpy as np

class Field(object):
    def __init__(self, name, shape=(1,), dtype=np.float64):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def compute_shape(self, n):
        if len(self.shape) == 1:
            return (n,) if self.shape[0] == 1 else (n, self.shape[0])
        else:
            raise Exception("Matrix types are not supported")

    def create_numpy(self, n):
        return np.zeros(self.compute_shape(n), self.dtype)

    def resize_numpy(self, a, n):
        a.resize(self.compute_shape(n), refcheck=False)


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


def create(cls_name, fields):

    view_props = {}
    for f in fields:
        def gen_property(name):
            return property(
                lambda self: getattr(self.soa, name)[self.id],
                lambda self, value: getattr(self.soa, name).__setitem__(self.id, value)
            )
        view_props[f.name] =  gen_property(f.name)
    
    def view_init(self, soa, id):
        SOAViewBase.__init__(self, soa, id)

    view_props['__init__'] = view_init
    
    view_cls = type(cls_name + 'View', (SOAViewBase,), view_props)

    def soa_init(self, capacity=0):
        SOABase.__init__(self, fields, capacity)

    soa_cls = type(cls_name, (SOABase,), {"__init__": soa_init})
    soa_cls.View = view_cls

    return soa_cls



MySOA = create('MySOA', fields=[
    Field('pos', dtype=np.float64, shape=(2,)),
    Field('vel', dtype=np.float64, shape=(2,)),
    Field('active', dtype=bool, shape=(1,)),
])

class Body(MySOA.View):
    def __init__(self, soa):
        super(Body, self).__init__(soa, soa.take())

    @property
    def modified_pos(self):
        return self.pos + 2


