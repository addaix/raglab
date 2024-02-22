"""Testing store utils"""

import os

from dol import temp_dir
from raglab.stores.stores_util import (
    mk_local_space_store,
    mk_json_store,
    mk_text_store,
    mk_pickle_store,
)


def test_mk_local_space_store(verbose=False):

    rootdir = temp_dir('ragdag/rjsf_fiddle')
    if verbose:
        print(f"{rootdir=}")

    s = mk_local_space_store(rootdir, space='bob', store_kind='spec_1')
    s['some_key'] = b'some_value'
    s['some_other_key'] = b'some_other_value'
    if verbose:
        print(f"Go check out this folder: file://{s.rootdir}")
    assert 'some_key' in s
    assert s['some_key'] == b'some_value'
    assert 'some_key' in os.listdir(s.rootdir)


def test_mk_specialized_local_space_store(verbose=False):

    rootdir = temp_dir('ragdag/rjsf_fiddle')

    if verbose:
        print(f"{rootdir=}")

    s = mk_json_store(rootdir, space='alice', store_kind='jsons')
    s['some_key'] = {'a': 'b'}
    assert s['some_key'] == {'a': 'b'}

    # see that the file is there, with a json extension
    assert 'some_key.json' in os.listdir(s.rootdir)
    # see that the file contains the expected content, as json string
    from pathlib import Path

    assert Path(os.path.join(s.rootdir, 'some_key.json')).read_text() == '{"a": "b"}'

    # showing how, when space is not provided, mk_text_store returns a store factory
    # with the rootdir fixed
    mk_text_store_for_user = mk_text_store(rootdir)
    s = mk_text_store_for_user('alice', store_kind='texts')
    s['some_key'] = 'some_value'
    assert s['some_key'] == 'some_value'

    # we can fix any attribute of the store factory
    mk_pickle_store_for_user = mk_pickle_store(rootdir, store_kind='pickles')
    s = mk_pickle_store_for_user(space='alice')
    ecclectic_data = [1, 2.3, 'four', map]
    s['some_key'] = ecclectic_data
    assert s['some_key'] == ecclectic_data
