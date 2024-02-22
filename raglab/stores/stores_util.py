"""Utils for stores"""

from typing import Callable, Optional
import os
from pathlib import Path
import json

from dol import (
    filt_iter,
    Files,
    KeyTemplate,
    Pipe,
    KeyCodecs,
    add_ipython_key_completions,
    mk_dirs_if_missing,
)
from dol import DirReader, wrap_kvs
from dol.filesys import with_relative_paths
from dol.util import not_a_mac_junk_path

pjoin = os.path.join

spaces_dirname = 'spaces'
spaces_template = pjoin(spaces_dirname, '{space}')
stores_template = pjoin('stores', '{store_kind}')
space_stores_template = pjoin(spaces_template, stores_template)


# TODO: Horrible way to do this, even for local stores!
# TODO: At the very least, make the store a write_back one
# TODO: Add better missing key (space) message through __missing__ method
def mk_space_info_store(rootdir):

    def _get_info_or_create_it(
        dir_where_info_is,
        *,
        info_filename='info.json',
        info_factory=lambda: {'kind': 'space_info'},
    ):
        info_filepath = os.path.join(dir_where_info_is, info_filename)
        if not os.path.exists(info_filepath):
            info_contents = info_factory()
            Path(info_filepath).write_text(json.dumps(info_contents))

        return json.loads(Path(info_filepath).read_text())

    space_dirpath = rootdir + f'/{spaces_dirname}'
    if not os.path.isdir(space_dirpath):
        os.makedirs(space_dirpath, exist_ok=True)
    s = wrap_kvs(
        DirReader(space_dirpath, max_levels=0),
        obj_of_data=lambda x: _get_info_or_create_it(x.rootdir),
    )
    s = with_relative_paths(s)
    s = KeyCodecs.suffixed(os.path.sep)(s)
    return s


# kt = KeyTemplate(template=space_stores_template)


def mk_local_space_store(
    rootdir,
    space: str = None,
    *,
    store_kind='miscellenous_stuff',
    rootdir_to_local_store: Callable = Files,
    rm_mac_junk=True,
    filename_suffix: str = '',
    filename_prefix: str = '',
    auto_make_dirs=True,
    key_autocomplete=True,
):
    _input_kwargs = locals()
    if not os.path.isdir(rootdir):
        raise ValueError(f"rootdir {rootdir} is not a directory")
    if space is None:
        # bind the rootdir, resulting in a function parametrized by space
        _input_kwargs = {
            k: v for k, v in _input_kwargs.items() if k not in {'rootdir', 'space'}
        }
        return partial(mk_local_space_store, rootdir, **_input_kwargs)
    assert space is not None, f"space must be provided"

    store_wraps = []
    if filename_suffix or filename_prefix:
        store_wraps.append(
            KeyCodecs.affixed(prefix=filename_prefix, suffix=filename_suffix)
        )
    if rm_mac_junk:
        store_wraps.append(filt_iter(filt=not_a_mac_junk_path))
    if auto_make_dirs:
        store_wraps.append(mk_dirs_if_missing)
        # if not os.path.isdir(rootdir):
        #     os.makedirs(rootdir, exist_ok=True)
    if key_autocomplete:
        store_wraps.append(add_ipython_key_completions)

    store_wrap = Pipe(*store_wraps)

    space_store_root = pjoin(
        rootdir,
        space_stores_template.format(space=space, store_kind=store_kind),
    )
    store = store_wrap(rootdir_to_local_store(space_store_root))
    return store


from functools import partial
import dill
from dol import TextFiles, JsonFiles, PickleFiles, wrap_kvs

mk_text_store = partial(
    mk_local_space_store, rootdir_to_local_store=TextFiles, filename_suffix='.txt'
)
mk_json_store = partial(
    mk_local_space_store, rootdir_to_local_store=JsonFiles, filename_suffix='.json'
)
mk_pickle_store = partial(
    mk_local_space_store, rootdir_to_local_store=PickleFiles, filename_suffix='.pkl'
)

# pickle is builtin, but fickle -- dill can serialize more things (lambdas, etc.)
LocalDillStore = wrap_kvs(Files, data_of_obj=dill.dumps, obj_of_data=dill.loads)
mk_dill_store = partial(
    mk_local_space_store, rootdir_to_local_store=LocalDillStore, filename_suffix='.dill'
)
