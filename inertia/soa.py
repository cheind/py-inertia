import numpy as np

class SOA(object):

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

    class View(object):
        def __init__(self, soa, id):
            self.__dict__['soa'] = soa
            self.__dict__['id'] = id
        
        def __getattr__(self, name):
            # Delegate to SOA here, in case it resizes we don't invalidate
            if hasattr(self.soa, name):
                return getattr(self.soa, name)[self.id]
            else:
                return super(View, self).__getattr__(name)

        def __setattr__(self, name, value):
            # Delegate to SOA here, in case it resizes we don't invalidate
            if hasattr(self.soa, name):
                getattr(self.soa, name)[self.id] = value
            else:
                super(View, self).__setattr__(name, value)

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
        return SOA.View(self, id)

    def __len__(self):
        return self.n


soa = SOA(fields=[
    SOA.Field('pos', dtype=np.float64, shape=(2,)),
    SOA.Field('vel', dtype=np.float64, shape=(2,)),
    SOA.Field('active', dtype=bool, shape=(1,)),
], capacity=2)


class Body(SOA.View):
    def __init__(self, soa):
        super(Body, self).__init__(soa, soa.take())

    @property
    def modified_pos(self):
        return self.pos + 2