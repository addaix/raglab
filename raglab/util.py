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
