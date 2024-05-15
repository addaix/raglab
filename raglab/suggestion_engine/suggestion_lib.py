import re
from meshed import DAG
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from functools import partial
from typing import List


def set_from_text(txt):
    """Process a text to extract the set of stack keywords"""
    to_remove = [
        "\n",
        "\\",
        " ",
        ",",
        ".",
        "(",
        ")",
        ":",
        ";",
        "!",
        "?",
        " de ",
        " en ",
    ]
    for r in to_remove:
        txt = txt.replace(r, " ")
    set_ = re.split(r"[,\s\n.]+", txt.lower())  # txt.split(', ')
    set_ = set(set_)
    if "" in set_:
        set_.remove("")
    return set_


# --------------------- Building the DAG ---------------------

from mlxtend.frequent_patterns import association_rules


def frequent_itemsets(
    transactions,
    min_support=0.1,
    # min_length=2,
    **kwargs,
):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True, **kwargs)
    # return frequent_itemsets with at least min_length items
    return freq_itemsets  # [freq_itemsets["itemsets"].apply(lambda x: len(x) >= min_length)]


def associations_table(
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


from concurrent.futures import ThreadPoolExecutor


def suggestions(
    associations_table: pd.DataFrame,
    new_itemset: set,
    predictive_keywords: set,
    min_confidence=0.6,
    min_lift=1,
):
    selected_rows = associations_table[
        (associations_table["antecedents"].apply(lambda x: x.issubset(new_itemset)))
        & (associations_table["confidence"] > min_confidence)
        & (associations_table["lift"] > min_lift)
    ]
    set_ = set()
    for consequents in selected_rows["consequents"]:
        set_.update(consequents)
    return set_.difference(predictive_keywords.union(new_itemset))


# def filtered_suggestions(
#     suggestions: set, new_itemset: set, min_confidence=0.6, min_lift=1
# ):
#     """returns a set og items that are suggested"""
#     filtered = set()
#     for _, row in suggestions.iterrows():
#         if row["confidence"] > min_confidence and row["lift"] > min_lift:
#             filtered.update(row["consequents"])
#     return filtered.difference(new_itemset)


funcs = [
    frequent_itemsets,
    associations_table,
    suggestions,
]
dag = DAG(funcs)
dag.dot_digraph()
