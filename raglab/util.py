"""Utils for raglab"""

import os
from config2py import get_configs_folder_for_app
from types import SimpleNamespace

app_dir = os.environ.get('RAGLAB_APP_FOLDER', None) or get_configs_folder_for_app(
    'raglab'
)


configs_dir = os.path.join(app_dir, 'configs')
data_dir = os.path.join(app_dir, 'data')
local_stores_dir = os.path.join(app_dir, 'local_stores')
users_dir_name = 'u'
local_space_stores_dir = os.path.join(local_stores_dir, users_dir_name)


for path in [configs_dir, data_dir, local_stores_dir]:
    os.makedirs(path, exist_ok=True)


class IterableNamespace(SimpleNamespace):
    def __iter__(self):
        yield from self.__dict__

    # make it read-only
    def __setattr__(self, name, value):
        raise AttributeError("can't set attribute")


from typing import MutableMapping


class StoreAccess:
    """
    Delegator for MutableMapping, providing list, read, write, and delete methods.

    This is intended to be used in web services, offering nicer method names than
    the MutableMapping interface, and an actual list instead of a generator in
    the case of list.
    """

    store: MutableMapping

    def list(self):
        return list(self.store.keys())

    def read(self, key):
        return self.store[key]

    def write(self, key, value):
        self.store[key] = value

    def delete(self, key):
        del self.store[key]


from i2 import call_forgivingly, Namespace
from typing import Mapping, Callable


class LazyAccessor(Namespace):
    """A namespace of factories, with lazy instantiation.

    Give it a `Mapping` of factory functions, and it will call these factories to make
    object instances only if and when they're requested
    (via `.attribute` or `[key]` access), keeping the instances created in a cache.

    Note that if the factories have parameters, they will be passed to the factories
    when the factories are called, via the `**named_parameters` argument.

    Note that if there are any parameters, any of them whose name matches the name of
    a factory will be used to instantiate the factory.

    >>> def plus_2(x):
    ...     return x + 2
    >>> def times(x, y):
    ...     return x * y
    >>> f = LazyAccessor(dict(plus_2=plus_2, times=times), x=10, y=2)
    >>> list(f)
    ['plus_2', 'times']
    >>> f.plus_2
    12
    >>> f.times
    20

    If you need to make a LazyAccessor factory with a fixed set of object factories,
    you can use `functools.partial` to do so:

    >>> from functools import partial
    >>> object_factories = dict(plus_2=plus_2, times=times)
    >>> mk_accessor = partial(LazyAccessor, object_factories)
    >>> g = mk_accessor(x=10, y=2)
    >>> g.plus_2, g.times
    (12, 20)
    >>> h = mk_accessor(x=3, y=10)
    >>> h.plus_2, h.times
    (5, 30)

    """

    def __init__(self, factories: Mapping[str, Callable], **named_parameters):
        self._factories = factories
        self._named_parameters = named_parameters
        self._cached_objs = {}

    def __getattr__(self, name):
        # Note: Could use a specialized "method" lru_cache instead.
        # See for example: https://stackoverflow.com/questions/33672412/python-functools-lru-cache-with-instance-methods-release-object/68052994#68052994
        if name not in self._cached_objs:
            obj = call_forgivingly(self._factories[name], **self._named_parameters)
            self._cached_objs[name] = obj
        return self._cached_objs[name]

    def __getitem__(self, k):
        return self._factories[k]

    def __iter__(self):
        return iter(self._factories)

    def __dir__(self):
        return list(self)
