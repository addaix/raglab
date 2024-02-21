"""Utils for stores"""

from typing import Callable, Optional
import os

import dill

from dol import filt_iter, Files, KeyTemplate, Pipe, KeyCodecs, add_ipython_key_completions, mk_dirs_if_missing
from dol.util import not_a_mac_junk_path

pjoin = os.path.join

user_template = pjoin('u', '{user_name}')
stores_template = pjoin('stores', '{store_kind}')
user_stores_template = pjoin(user_template, stores_template)

kt = KeyTemplate(template=user_stores_template)

def mk_local_user_store(
        rootdir,
          *, 
        user_name='catchall', 
        store_kind='misc_binaries', 
        rootdir_to_local_store: Callable = Files, 
        rm_mac_junk=True,
        filename_suffix: str = '',
        filename_prefix: str = '',
        auto_make_dirs=True,
        key_autocomplete=True,
    ):
    store_wraps = []
    if filename_suffix or filename_prefix:
        store_wraps.append(
            KeyCodecs.affixed(prefix=filename_prefix, suffix=filename_suffix)
        )
    if rm_mac_junk:
        store_wraps.append(filt_iter(filt=not_a_mac_junk_path))
    if auto_make_dirs:
        store_wraps.append(mk_dirs_if_missing)
    if key_autocomplete:
        store_wraps.append(add_ipython_key_completions)
    
    store_wrap = Pipe(*store_wraps)

    user_store_root = pjoin(
        rootdir,
        user_stores_template.format(
            user_name=user_name, store_kind=store_kind
        )
    )
    store = store_wrap(rootdir_to_local_store(user_store_root))
    return store

from functools import partial
from dol import TextFiles, JsonFiles, PickleFiles, wrap_kvs

mk_text_store = partial(mk_local_user_store, rootdir_to_local_store=TextFiles, filename_suffix='.txt')
mk_json_store = partial(mk_local_user_store, rootdir_to_local_store=JsonFiles, filename_suffix='.json')
mk_pickle_store = partial(mk_local_user_store, rootdir_to_local_store=PickleFiles, filename_suffix='.pkl')
LocalDillStore = wrap_kvs(Files, data_of_obj=dill.dumps, obj_of_data=dill.loads)
