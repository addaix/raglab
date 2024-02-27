"""Data Driven Execution for Prompt Templates"""

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
    add_ipython_key_completions,
)
from oa import prompt_function
from ju import func_to_form_spec


def mk_prompt_template_store(
    *,
    space='prompt_tests',
    store_kind='prompt_templates',
    data_dir=data_dir,
    prepopulate_with_some_templates=True
):
    txt_filt = filt_iter.suffixes('.txt')
    txt = KeyCodecs.suffixed('.txt')
    prompt_template_wrap = Pipe(txt_filt, txt)

    LocalPromptTemplates = Pipe(
        TextFiles, mk_dirs_if_missing, prompt_template_wrap, add_ipython_key_completions
    )

    # Make a store
    rootdir = os.path.join(
        data_dir, space_stores_template.format(space=space, store_kind=store_kind)
    )
    prompt_templates = LocalPromptTemplates(rootdir)

    if prepopulate_with_some_templates:
        # pre_populate it with some templates
        from oa.util import chatgpt_templates_dir

        src = prompt_template_wrap(TextFiles(chatgpt_templates_dir))
        prompt_templates.update(src)
        list(prompt_templates)

    return prompt_templates


def _default_chat():
    from oa import chat

    return chat


class PromptDDE(StoreAccess):

    def __init__(self, store, chat=None):
        self.store = store
        self.chat = chat or _default_chat()

    def prompt_func(self, prompt_template: str, **kwargs):
        return prompt_function(prompt_template, prompt_func=self.chat, **kwargs)

    def execute_prompt(self, prompt_template: str, params: dict, **prompt_func_kwargs):
        func = self.prompt_func(prompt_template, **prompt_func_kwargs)
        return func(**params)

    def execute_prompt_from_key(self, key: str, params: dict, **prompt_func_kwargs):
        prompt_template = self.read(key)
        func = self.prompt_func(prompt_template, **prompt_func_kwargs)
        return func(**params)

    def rjsf_json_of_prompt_template(self, prompt_template: str, **prompt_func_kwargs):
        return func_to_form_spec(
            self.prompt_func(prompt_template, **prompt_func_kwargs)
        )

    def rjsf_json_of_key(self, key: str, **prompt_func_kwargs):
        prompt_template = self.read(key)
        return func_to_form_spec(
            self.prompt_func(prompt_template, **prompt_func_kwargs)
        )



prompt_template_dde = PromptDDE(mk_prompt_template_store())


handlers = [
    {
        'endpoint': prompt_template_dde,
        'name': 'prompt_templates',
        'attr_names': [
            'list',
            'read',
            'write',
            'delete',
            'execute_prompt',
            'execute_prompt_from_key',
            'rjsf_json_of_prompt_template',
            'rjsf_json_of_key'
        ],
    }
]


if __name__ == '__main__':
    from py2http import run_app

    run_app(handlers, publish_openapi=True, publish_swagger=True)
