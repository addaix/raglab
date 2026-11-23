"""Tests for the cross-run benchmark (raglab#15, item 4). Hermetic: fakes only."""

from ir import SearchHit

from raglab import PurposeStore
from raglab.bench import cross_run_benchmark


def _hits(*ids, source="s"):
    return [SearchHit(aid, "k", 1.0 - i / 10, f"text {aid}", {}, source) for i, aid in enumerate(ids)]


def _static(*ids):
    return lambda query, **kw: _hits(*ids)


VOCAB = ("invoice", "invoices", "2023", "cats")


def _bag_embedder(text):
    words = text.split()
    return [float(words.count(w)) + (w == "invoice") * words.count("invoices") for w in VOCAB]


def test_repeats_are_served_from_memory_with_nothing_lost():
    report = cross_run_benchmark({"s": _static("a", "b")}, ["q", "q", "q", "cats"])
    assert [r.sweeps_cold for r in report.rows] == [1, 1, 1, 1]
    assert [r.sweeps_warm for r in report.rows] == [1, 0, 0, 1]
    assert report.sweeps_saved == 0.5
    assert report.recall == 1.0
    assert report.extractions == {"q": 2, "cats": 2}


def test_paraphrases_share_findings_only_with_an_embedder():
    workload = ["invoices 2023", "2023 invoices"]
    plain = cross_run_benchmark({"s": _static("a")}, workload)
    embedded = cross_run_benchmark(
        {"s": _static("a")}, workload, memory_factory=lambda: PurposeStore(embedder=_bag_embedder)
    )
    assert plain.sweeps_warm == 2
    assert embedded.sweeps_warm == 1


def test_source_mutation_shows_as_lost_recall():
    # the SUPERSEDE trigger: the source changed, memory still serves the old hit
    calls = []

    def mutating(query, **kw):
        calls.append(query)
        # the first purpose is swept by both runs, then the source moves on
        return _hits("a" if len(calls) <= 2 else "z")

    report = cross_run_benchmark({"s": mutating}, ["q", "q"])
    assert [r.recall for r in report.rows] == [1.0, 0.0]
    assert report.recall == 0.5


def test_memory_is_the_benchmarks_to_set():
    import pytest

    with pytest.raises(TypeError, match="sets memory itself"):
        cross_run_benchmark({"s": _static("a")}, ["q"], memory=PurposeStore())


def test_extractions_count_each_purpose_own_units_even_with_an_embedder():
    report = cross_run_benchmark(
        {"s": _static("a")},
        ["invoices 2023", "2023 invoices"],
        memory_factory=lambda: PurposeStore(embedder=_bag_embedder),
    )
    assert report.extractions == {"invoices 2023": 1}


# --- loop vs single discover (raglab#20) ---------------------------------------------


def test_loop_vs_single_counts_the_extra_calls_and_the_hits_only_the_loop_finds():
    from raglab.agent import Judgement, SubTask
    from raglab.bench import loop_vs_single_benchmark

    def retrieve(query, **kw):
        return _hits("a", "b") if query == "q" else _hits("c")

    def evaluator(task, results):
        if task.goal == "q":  # first round: not enough, re-query with a refinement
            return Judgement(relevant=results, sufficient=False, refinement=SubTask("q2", task.sources))
        return Judgement(relevant=results, sufficient=True)

    report = loop_vs_single_benchmark(
        {"s": retrieve}, ["q"], relevant={"q": ["a", "c"]}, evaluator=evaluator
    )
    (row,) = report.rows
    assert (row.calls_single, row.calls_loop) == (1, 2)
    assert report.extra_calls == 2.0
    assert row.hits_single == 2 and row.hits_loop >= 3
    assert row.single_recall_of_loop < 1.0
    assert report.recall_single == 0.5 and report.recall_loop == 1.0


def test_loop_vs_single_with_no_loop_roles_is_identical():
    from raglab.bench import loop_vs_single_benchmark

    report = loop_vs_single_benchmark({"s": _static("a", "b")}, ["q1", "q2"])
    assert report.extra_calls == 1.0 and report.single_recall_of_loop == 1.0
    assert report.recall_loop is None
