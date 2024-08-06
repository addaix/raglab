"""Data Driven Execution for functions"""

import os
from raglab.stores.stores_util import space_stores_template
from raglab.util import data_dir
from raglab.util import StoreAccess
from dol import (
    TextFiles,
    filt_iter,
    KeyCodecs,
    Pipe,
    mk_dirs_if_missing,
)
from oa import prompt_function


def mk_function_store_pipe(
    *,
    space="functions_tests",
    store_kind="functions",
    data_dir=data_dir,
):
    filter = filt_iter.suffixes(".json")
    codec = KeyCodecs.suffixed(".json")
    filter_pipe = Pipe(filter, codec)

    stored_functions = Pipe(TextFiles, mk_dirs_if_missing, filter_pipe)

    # Make a store
    rootdir = os.path.join(
        data_dir, space_stores_template.format(space=space, store_kind=store_kind)
    )
    store_pipe = stored_functions(rootdir)

    return store_pipe


def _default_chat():
    from oa import chat

    return chat


class FunctionStore(StoreAccess):

    def __init__(self, store):
        self.store = store


prompt_template_dde = FunctionStore(mk_function_store_pipe())


handlers = [
    {
        "endpoint": prompt_template_dde,
        "name": "prompt_templates",
        "attr_names": [
            "list",
            "read",
            "write",
            "delete",
            "execute_prompt",
            "execute_prompt_from_key",
        ],
    }
]


if __name__ == "__main__":
    from py2http import run_app

    run_app(handlers, publish_openapi=True, publish_swagger=True)
