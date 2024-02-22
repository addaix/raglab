"""Simple stores for raglab"""

from dataclasses import dataclass
from functools import partial, lru_cache
import os
from raglab.stores.stores_util import (
    mk_json_store,
    mk_text_store,
    mk_dill_store,
    spaces_dirname,
)
from raglab.util import local_stores_dir, IterableNamespace

DFLT_LRU_CACHE_SIZE = 128


def _set_store_kind(d: dict):
    return {k: partial(v, store_kind=k) for k, v in d.items()}


space_store_factories = {
    'user_preferences': mk_json_store,
    'rjsf_forms': mk_json_store,
    'datasets': mk_json_store,
    'prompt_templates_text': mk_text_store,
    'py_objects': mk_dill_store,
}
space_store_factories = _set_store_kind(space_store_factories)


# @lru_cache(maxsize=DFLT_LRU_CACHE_SIZE)
def mk_user_mall(space, local_stores_rootdir: str = local_stores_dir):
    return {
        name: factory(local_stores_rootdir, space=space)
        for name, factory in space_store_factories.items()
    }


def mk_user_mall_namespace(local_stores_rootdir: str = local_stores_dir):
    mall = mk_user_mall(local_stores_rootdir=local_stores_rootdir)
    return IterableNamespace(**mall)


class UserLocalMalls:
    def __init__(self, local_stores_rootdir: str = local_stores_dir):
        self.local_stores_rootdir = os.path.abspath(local_stores_rootdir)
        self.space_stores_rootdir = os.path.join(
            self.local_stores_rootdir, spaces_dirname
        )

    def __iter__(self):
        yield from os.path.listdir(self.space_stores_rootdir)

    def __getitem__(self, space):
        return mk_user_mall(space, local_stores_rootdir=self.local_stores_rootdir)

    def __setitem__(self, space, mall):
        pass

    def __len__(self, space, mall):
        return len(list(self))

    def __contains__(self, space):
        return os.path.isdir(os.path.join(self.space_stores_rootdir, space))
