"""Cross-run benchmark: the instrument for purpose-indexed memory (raglab#15, item 4).

The follow-ups of the memory design (CONSOLIDATE, SUPERSEDE, purpose transfer, DECAY)
are each gated on an *observed* trigger. This module is what observes: it replays a
workload of purposes through an agent twice, once cold (no memory) and once warm (with
a :class:`~raglab.memory.PurposeStore`), counting the source calls of each, and reports

* ``sweeps_cold`` / ``sweeps_warm``: retriever calls without and with memory (what
  memory saves),
* ``recall``: the share of the cold run's hits the warm run also returned (what memory
  costs in answer quality: ``1.0`` means nothing was lost),
* ``extractions``: units kept per purpose after the warm run (the CONSOLIDATE
  trigger is "this exceeds a threshold").

Cold and warm calls interleave on the same sources, purpose by purpose, so a source
that changes over time shows as lost recall: that is the SUPERSEDE trigger, but so is
plain non-determinism (an approximate index, a live API). Run the benchmark on a
deterministic snapshot to read ``recall`` as what memory alone lost.

Hermetic and offline: pass any sources; nothing here calls an LLM::

    report = cross_run_benchmark(sources, ["invoices 2023", "2023 invoices"])
    report.sweeps_saved, report.recall
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .agent import Budget, Retriever, make_search_agent
from .memory import PurposeStore, purpose_id

AgentFactory = Callable[..., Callable[[str], list]]


@dataclass(frozen=True)
class PurposeRow:
    """One purpose of the workload: its source calls cold and warm, and its recall."""

    purpose: str
    sweeps_cold: int
    sweeps_warm: int
    recall: float


@dataclass(frozen=True)
class CrossRunReport:
    """Totals over a workload, with the per-purpose ``rows`` and ``extractions``."""

    rows: tuple[PurposeRow, ...]
    extractions: Mapping[str, int] = field(default_factory=dict)

    @property
    def sweeps_cold(self) -> int:
        return sum(r.sweeps_cold for r in self.rows)

    @property
    def sweeps_warm(self) -> int:
        return sum(r.sweeps_warm for r in self.rows)

    @property
    def sweeps_saved(self) -> float:
        """Share of the cold source calls the warm run did not make."""
        return 1 - self.sweeps_warm / self.sweeps_cold if self.sweeps_cold else 0.0

    @property
    def recall(self) -> float:
        """Mean, over purposes, of the share of cold hits the warm run also returned."""
        return sum(r.recall for r in self.rows) / len(self.rows) if self.rows else 1.0


def extraction_counts(store: PurposeStore) -> dict[str, int]:
    """``{purpose: number of units kept}`` (each purpose under its own statement).

    >>> from raglab.memory import ExtractionUnit
    >>> s = PurposeStore()
    >>> s["q"] = [ExtractionUnit("q", {"artifact_id": "a"}, 0.0)]
    >>> extraction_counts(s)
    {'q': 1}
    """
    return {
        purpose: len(
            store._units(purpose_id(purpose))
        )  # its own units, not related ones
        for purpose in store
    }


class _Counting(Mapping):
    """``sources``, lazily, with each retriever call appended to ``calls``."""

    def __init__(self, sources: Mapping[str, Retriever], calls: list):
        self._sources, self._calls = sources, calls

    def __getitem__(self, name: str) -> Retriever:
        retrieve = self._sources[name]

        def counted(*args, **kwargs):
            self._calls.append(name)
            return retrieve(*args, **kwargs)

        return counted

    def __iter__(self):
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


def _hit_key(hit: Any) -> tuple:
    return (getattr(hit, "source", None), hit.artifact_id)


def cross_run_benchmark(
    sources: Mapping[str, Retriever],
    workload: Iterable[str],
    *,
    agent_factory: AgentFactory = make_search_agent,
    memory_factory: Callable[[], PurposeStore] = PurposeStore,
    **agent_kwargs: Any,
) -> CrossRunReport:
    """Replay ``workload`` (purposes, in order, repeats and paraphrases welcome) cold
    and warm, and report what memory saved and what it lost.

    ``agent_factory(sources, memory=..., **agent_kwargs)`` builds the agent (default
    :func:`~raglab.agent.make_search_agent`); the cold agent gets ``memory=None``.
    The warm agent shares one ``memory_factory()`` store across the whole workload.
    """
    if "memory" in agent_kwargs:
        raise TypeError(
            "cross_run_benchmark sets memory itself (none cold, memory_factory() warm)"
        )
    workload = list(workload)
    cold_calls: list = []
    cold = agent_factory(_Counting(sources, cold_calls), memory=None, **agent_kwargs)
    warm_calls: list = []
    store = memory_factory()
    warm = agent_factory(_Counting(sources, warm_calls), memory=store, **agent_kwargs)

    rows = []
    for purpose in workload:
        n_cold, n_warm = len(cold_calls), len(warm_calls)
        cold_hits = {_hit_key(h) for h in cold(purpose)}
        warm_hits = {_hit_key(h) for h in warm(purpose)}
        recall = len(cold_hits & warm_hits) / len(cold_hits) if cold_hits else 1.0
        rows.append(
            PurposeRow(
                purpose,
                sweeps_cold=len(cold_calls) - n_cold,
                sweeps_warm=len(warm_calls) - n_warm,
                recall=recall,
            )
        )
    return CrossRunReport(tuple(rows), extraction_counts(store))


# --------------------------------------------------------------------------- #
# Loop vs single discover (raglab#20)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionRow:
    """One question: source calls and hits of the loop and of a single discover."""

    question: str
    calls_loop: int
    calls_single: int
    hits_loop: int
    hits_single: int
    #: share of the loop's hits that a single discover also returned (1.0: the loop added nothing)
    single_recall_of_loop: float
    #: with ``relevant``: share of the relevant ids each variant returned (else ``None``)
    recall_loop: float | None = None
    recall_single: float | None = None


@dataclass(frozen=True)
class LoopVsSingleReport:
    """Totals over the questions, with the per-question ``rows``."""

    rows: tuple[QuestionRow, ...]

    @property
    def calls_loop(self) -> int:
        return sum(r.calls_loop for r in self.rows)

    @property
    def calls_single(self) -> int:
        return sum(r.calls_single for r in self.rows)

    @property
    def extra_calls(self) -> float:
        """Source calls the loop makes per call of a single discover (``1.0``: no extra)."""
        return self.calls_loop / self.calls_single if self.calls_single else 0.0

    @property
    def single_recall_of_loop(self) -> float:
        """Mean share of the loop's hits a single discover also returned."""
        n = len(self.rows)
        return sum(r.single_recall_of_loop for r in self.rows) / n if n else 1.0

    def _mean(self, name: str) -> float | None:
        values = [getattr(r, name) for r in self.rows if getattr(r, name) is not None]
        return sum(values) / len(values) if values else None

    @property
    def recall_loop(self) -> float | None:
        return self._mean("recall_loop")

    @property
    def recall_single(self) -> float | None:
        return self._mean("recall_single")


def loop_vs_single_benchmark(
    sources: Mapping[str, Retriever],
    questions: Iterable[str],
    *,
    relevant: Mapping[str, Iterable[str]] | None = None,
    agent_factory: AgentFactory = make_search_agent,
    single_budget: Budget | None = None,
    **loop_kwargs: Any,
) -> LoopVsSingleReport:
    """Run each question through the agent loop and through a single discover; compare.

    The *single* run is the same agent with ``single_budget`` (default one round, one
    pass over the sources) and none of ``loop_kwargs``: no LLM roles, so no rewriting
    and no back-edge. The *loop* run gets ``loop_kwargs`` (``formulator``, ``evaluator``,
    ``planner``, ``budget``...). Both count their retriever calls. ``relevant`` maps a
    question to the artifact ids that answer it, and adds recall against that ground
    truth; without it the report says how much of the loop's answer a single discover
    already had, and what the loop costs in extra source calls. Offline when the roles
    are: pass fakes, or live sources for a real measurement.
    """
    single_budget = single_budget or Budget(max_rounds=1)
    rows = []
    for question in list(questions):
        loop_calls: list = []
        single_calls: list = []
        loop = agent_factory(_Counting(sources, loop_calls), memory=None, **loop_kwargs)
        single = agent_factory(
            _Counting(sources, single_calls), memory=None, budget=single_budget
        )
        loop_hits = {_hit_key(h) for h in loop(question)}
        single_hits = {_hit_key(h) for h in single(question)}
        shared = len(loop_hits & single_hits) / len(loop_hits) if loop_hits else 1.0
        truth = None if relevant is None else set(relevant.get(question, ()))
        rows.append(
            QuestionRow(
                question,
                calls_loop=len(loop_calls),
                calls_single=len(single_calls),
                hits_loop=len(loop_hits),
                hits_single=len(single_hits),
                single_recall_of_loop=shared,
                recall_loop=_recall(loop_hits, truth),
                recall_single=_recall(single_hits, truth),
            )
        )
    return LoopVsSingleReport(tuple(rows))


def _recall(hits: set, truth: set | None) -> float | None:
    """Share of the ``truth`` ids among ``hits`` (``None`` without ground truth)."""
    if not truth:
        return None
    return len({artifact_id for _, artifact_id in hits} & truth) / len(truth)
