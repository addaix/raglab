"""Mockup webservices"""

import re
from typing import Mapping

import sys

# from srag.tw.tw_util import _module_callables, _skip_this


# @_skip_this
# def igore_this(x):
#     return x + 1


def user_function(obj):
    obj._user_function__ = True
    return obj


def is_user_function(obj):
    return hasattr(obj, "_user_function__")


def get_user_functions(module):
    return [v for k, v in vars(module).items() if is_user_function(v)]


datasets = {
    'philosophy': {
        "statement_1": "Bob is a man",
        "statement_2": "Bob is a name",
        "statement_3": "Men are mortal",
        "statement_4": "1 + 1 = 2",
    },
    'fruit': {
        "apples": "Apples are red",
        "more_apples": "Apples are fruit",
        "good": "Fruit are good",
        "outlier": "Cars drive on the road",
    },
}


@user_function
def dataset_list():
    return list(datasets.keys())


stopwords = {
    'is',
    'are',
    'a',
    'an',
    'the',
    'on',
    'in',
    'at',
    'for',
    'to',
    'of',
    'and',
    'or',
    'not',
    'no',
    'yes',
    'true',
    'false',
    'good',
    'bad',
}


@user_function
def add_stopword(word):
    stopwords.add(word.lower())


@user_function
def get_stopwords():
    return list(stopwords)


@user_function
def search_dataset(
    dataset_name: str,
    query: str,
    # *,
    # datasets: Mapping = datasets,
    # stopwords: set = stopwords,
):
    """Searches the dataset given by dataset_name for content matching query"""
    query_terms = map(lambda x: x.lower(), re.findall(r'\w+', query))
    query_terms = [x for x in query_terms if x not in stopwords]
    query_terms_pattern = re.compile(
        '|'.join(f"({x})" for x in query_terms), re.IGNORECASE
    )

    query_matches = lambda x: query_terms_pattern.search(x) is not None

    dataset = datasets[dataset_name]

    def matching_dataset_ids():
        for key, contents in dataset.items():
            if query_matches(contents):
                yield key

    return list(matching_dataset_ids())


# _current_module = sys.modules[__name__]

# funcs = [dataset_list, add_stopword, get_stopwords, search_dataset]
# funcs = [dataset_list, add_stopword, get_stopwords]

import sys

_current_module = sys.modules[__name__]
funcs = get_user_functions(_current_module)


if __name__ == '__main__':
    from py2http import run_app

    run_app(funcs, publish_openapi=True, publish_swagger=True)

    # from py2http import mk_app

    # app = mk_app(funcs, publish_openapi=True, publish_swagger=True)

    # from waitress import serve as run_app
    # run_app(app)

    # from flask import Flask

    # Flask(__name__).run(app)

# if __name__ == '__main__':

#     from streamlitfront import mk_app

#     app = mk_app(funcs)
#     app()
