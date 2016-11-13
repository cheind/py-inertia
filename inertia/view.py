
from enum import Enum

class LazyViewBase(object):
    """The base class for lazy views on numpy structured arrays."""

    def __init__(self, arr, slice):
        self.arr = arr
        self.slice = slice

def create_view(cls_name, fields):
    """Returns a lazy view class for the given SOA fields."""

    the_view = {}
    
    def init(self, arr, slice):
        LazyViewBase.__init__(self, arr, slice)
    the_view['__init__'] = init
    
    if issubclass(fields, Enum):
        fields = [e.value for e in fields]

    # Attach a r/w property for each field
    for f in fields:
        def gen_property(name):
            return property(
                lambda self: self.arr[self.slice][name],
                lambda self, value: self.arr[self.slice].__setitem__(name, value)
            )
        the_view[f[0]] = gen_property(f[0])
    
    view_cls = type(cls_name, (LazyViewBase,), the_view)
    return view_cls
