"""Webservices for store access"""

from functools import lru_cache
from typing import Any, MutableMapping
from dol import wrap_kvs
from raglab.stores.simple_stores import mk_user_mall
from raglab.util import StoreAccess


DFLT_LRU_CACHE_SIZE = 128


@lru_cache(maxsize=DFLT_LRU_CACHE_SIZE)
def get_user_mall(space: str) -> MutableMapping:
    mall = mk_user_mall(space)
    _mall = wrap_kvs(mall, obj_of_data=StoreAccess)
    return StoreAccess(_mall)


def get_user_mall_mall(space) -> MutableMapping:
    return get_user_mall(space).store


from raglab.util import local_stores_dir
from raglab.stores.stores_util import mk_space_info_store

space_info_store = mk_space_info_store(local_stores_dir)


handlers = [
    dict(
        endpoint=space_info_store,
        name='space_info',
        attr_names=['__iter__', '__getitem__', '__setitem__'],
    ),
]


if __name__ == '__main__':
    # from py2http import mk_app
    # app = mk_app(handlers, publish_openapi=True, publish_swagger=True)
    # app.run()

    from py2http import run_app

    run_app(handlers, publish_openapi=True, publish_swagger=True)
