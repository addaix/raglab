from concurrent.futures import ThreadPoolExecutor
from functools import partial
from meshed import DAG
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import pandas as pd
from random import randint
import re
from smart_cv import mall
from typing import List, ItemsView


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


jobs_db = set_from_text(mall.stack_mining["job_titles.txt"])
stack_db = set_from_text(mall.stack_mining["stacks.txt"])

full_stack_db = stack_db.union(jobs_db)


def transactions(cvs: ItemsView, stacks: set) -> list[set]:
    """Find the intersection between the stacks and the cvs.
    Args:
        cvs: Iterable of cvs names
        stacks: set of stack keywords
    Returns:
        list of sets
    """

    intersections = []
    for cv_name, cv_text in cvs:
        cv_text = mall.cvs[cv_name]

        stack_cv = set_from_text(cv_text)
        intersection = stack_cv.intersection(stacks)
        intersections.append(intersection)
    return intersections


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
    predictive_keywords: set = jobs_db,
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


def confidence_filter(
    frequent_itemsets: pd.DataFrame, associations_rule, confidence_threshold
):
    """Keep only the suggestions with a confidence above the threshold"""
    assert isinstance(associations_rule, tuple), "associations_rule must be a tuple"
    sub_itemset, suggestion = associations_rule
    c = next(confidence(frequent_itemsets, sub_itemset, suggestion))
    if c > confidence_threshold:
        return suggestion
    return set()


from concurrent.futures import ThreadPoolExecutor


def confidence_filtered_suggestion(
    freq_itemsets: pd.DataFrame,
    associations_rules,
    confidence_threshold,
):
    """Keep only the suggestions with a confidence above the threshold"""
    associations_rules_list = list(associations_rules)
    with ThreadPoolExecutor() as executor:
        res = executor.map(
            partial(
                confidence_filter,
                freq_itemsets,
                confidence_threshold=confidence_threshold,
            ),
            associations_rules_list,
        )
    return {item for frozenset_item in res for item in frozenset_item}


def suggestions(associations_table, new_itemset, predictive_keywords: set = jobs_db):
    return associations_table[
        (
            associations_table["consequents"].apply(
                lambda x: x not in new_itemset.union(predictive_keywords)
            )
        )
        & (
            associations_table["antecedents"].apply(
                lambda x: x.issubset(new_itemset) and x not in new_itemset
            )
        )
    ]


def filtered_suggestions(suggestions, min_confidence=0.6, min_lift=1):
    """returns a set og items that are suggested"""
    filtered = set()
    for _, row in suggestions.iterrows():
        if row["confidence"] > min_confidence and row["lift"] > min_lift:
            filtered.update(row["consequents"])
    return filtered


funcs = [
    frequent_itemsets,
    associations_table,
    suggestions,
    filtered_suggestions,
    # confidence_filtered_suggestion,
    transactions,
]
dag = DAG(funcs)
dag.dot_digraph()
