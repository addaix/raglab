"""Testing store utils"""

from dol import temp_dir
from raglab.stores.stores_util import (
    mk_local_user_store,
    mk_json_store,
    mk_text_store,
    mk_pickle_store,
)


def test_mk_local_user_store(verbose=False):

    rootdir = temp_dir('ragdag/rjsf_fiddle')
    if verbose:
        print(f"{rootdir=}")

    s = mk_local_user_store(rootdir, user_name='bob', store_kind='spec_1')
    s['some_key'] = b'some_value'
    s['some_other_key'] = b'some_other_value'
    if verbose:
        print(f"Go check out this folder: file://{s.rootdir}")
    assert 'some_key' in s
    assert s['some_key'] == b'some_value'
    assert 'some_key' in os.listdir(s.rootdir)


def test_mk_specialized_local_user_store(verbose=False):

    rootdir = temp_dir('ragdag/rjsf_fiddle')
    if verbose:
        print(f"{rootdir=}")

    s = mk_json_store(rootdir, user_name='alice', store_kind='jsons')
    s['some_key'] = {'a': 'b'}
    assert s['some_key'] == {'a': 'b'}

    # see that the file is there, with a json extension
    assert 'some_key.json' in os.listdir(s.rootdir)
    # see that the file contains the expected content, as json string
    from pathlib import Path

    assert Path(os.path.join(s.rootdir, 'some_key.json')).read_text() == '{"a": "b"}'

    s = mk_text_store(rootdir, user_name='alice', store_kind='texts')
    s['some_key'] = 'some_value'
    assert s['some_key'] == 'some_value'

    s = mk_pickle_store(rootdir, user_name='alice', store_kind='pickles')
    ecclectic_data = [1, 2.3, 'four']
    s['some_key'] = ecclectic_data
    assert s['some_key'] == ecclectic_data
