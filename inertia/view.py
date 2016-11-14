
from enum import Enum

class StructuredViewBase(object):
    """The base class for lazy views on numpy structured arrays."""

    def __init__(self, arr, slice, lazy=None):
        self.arr = arr
        self.slice = slice
        self.lazy = lazy

def create_view(cls_name, fields, lazy=True):
    """Returns a lazy view class for the given SOA fields."""

    the_view = {}
    
    if lazy:
        def init(self, arr, slice):
            StructuredViewBase.__init__(self, arr, slice, lazy=True)
        the_view['__init__'] = init
    else:
        def init(self, arr, slice):
            StructuredViewBase.__init__(self, arr[slice], slice, lazy=False)
        the_view['__init__'] = init

    if issubclass(fields, Enum):
        fields = [e.value for e in fields]

    # Attach a r/w property for each field
    for f in fields:
        def gen_property_lazy(name):
            return property(
                lambda self: self.arr[self.slice][name],
                lambda self, value: self.arr[self.slice].__setitem__(name, value)
            )

        def gen_property(name):
            return property(
                lambda self: self.arr[name],
                lambda self, value: self.arr.__setitem__(name, value)
            )
        

        the_view[f[0]] = gen_property_lazy(f[0]) if lazy else gen_property(f[0])
    
    view_cls = type(cls_name, (StructuredViewBase,), the_view)
    return view_cls
