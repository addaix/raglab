""" This module contains functions agregated in a DAG that does :
- extracts sets of words in a text
- match the sets of words with a set of keywords
- find associations between sets of words
- suggest new keywords based on the associations
"""

from concurrent.futures import ThreadPoolExecutor
import re
from meshed import DAG
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import numpy as np
from functools import partial
from typing import List


# --------------------- Extracting sets of words ---------------------
def set_from_text_maj(txt):
    """Process a text to extract the set of stack keywords"""
    to_remove = [
        "\n",
        "\\",
        "[",
        "]",
        '"',
        "{",
        "}",
        " ",
        ",",
        ".",
        "(",
        ")",
        ":",
        ";",
        "!",
        "?",
        " '",
        "' ",
        " de ",
        " en ",
        " le ",
        " la ",
        " les ",
    ]
    for r in to_remove:
        txt = txt.replace(r, " ")
    txt = re.sub(r"[^a-zA-Z]'|'[^a-zA-Z]|[^a-zA-Z]\"|\"[^a-zA-Z]", " ", txt)
    set_ = re.split(r"[,\s\n.]+", txt)  # txt.split(', ')
    set_ = set(set_)
    if "" in set_:
        set_.remove("")
    return set_


def set_from_text(txt):
    return set([i.lower() for i in set_from_text_maj(txt)])


def matching_keywords(text, keywords_set):
    lower_mapping = {i.lower(): i for i in keywords_set}
    text_set = set_from_text(text)
    intersection = text_set.intersection(lower_mapping)
    return set([lower_mapping[i] for i in intersection])


# --------------------- Building the DAG ---------------------

from mlxtend.frequent_patterns import association_rules


def frequent_itemsets(
    transactions,
    min_support=0.1,
    **kwargs,
):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, **kwargs)
    # return frequent_itemsets with at least min_length items
    return freq_itemsets  # [freq_itemsets["itemsets"].apply(lambda x: len(x) >= min_length)]


def associations_table_filtered(
    frequent_itemsets: pd.DataFrame,
    predictive_keywords: set,
    metric="confidence",
    min_threshold=0.5,
):
    """return the association table with the metric and the min_threshold as filters.
    return only rows whose consequent is not in the predictive_keywords set"""
    candidate_table = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold
    )
    return candidate_table[
        candidate_table["consequents"].apply(lambda x: x not in predictive_keywords)
    ]


def associations_table(
    transactions,
    previous_transactions=[],
    min_support=0.1,
    predictive_keywords: set = set(),
    metric="confidence",
    min_threshold=0.5,
    **kwargs,
):
    """return the association table with the metric and the min_threshold as filters."""
    return associations_table_filtered(
        frequent_itemsets=frequent_itemsets(
            transactions + previous_transactions, min_support=min_support, **kwargs
        ),
        predictive_keywords=predictive_keywords,
        metric=metric,
        min_threshold=min_threshold,
    )


def suggestions(
    associations_table: pd.DataFrame,
    new_itemset: set,
    predictive_keywords: set = set(),
    min_confidence=0.6,
    min_lift=1,
):
    """returns a set of words that are suggested based on the associations table, predictive_keywords, min_confidence and min_lift
    Args:
        associations_table : pd.DataFrame : the association table
        new_itemset : set : the new itemset
        predictive_keywords : set : the set of predictive keywords (words that help for suggestion but are not suggested)
        min_confidence : float : the minimum confidence
        min_lift : float : the minimum lift

        Returns:
            set : the set of suggested words"""
    selected_rows = associations_table[
        (associations_table["antecedents"].apply(lambda x: x.issubset(new_itemset)))
        & (associations_table["confidence"] > min_confidence)
        & (associations_table["lift"] > min_lift)
    ]
    set_ = set()
    for consequents in selected_rows["consequents"]:
        set_.update(consequents)
    return set_.difference(predictive_keywords.union(new_itemset))


funcs = [
    frequent_itemsets,
    associations_table,
    suggestions,
]
dag = DAG(funcs)
dag.dot_digraph()
