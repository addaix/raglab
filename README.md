# raglab

The **agentic-search / RAG orchestration layer on top of [`ir`](https://github.com/i2mint/ir)** (the retrieval substrate).

`raglab` turns a retrieval substrate into a **Composable Search Agent**: a small
set of injected **roles** — Planner, Formulator, Retriever, Evaluator, Reranker,
Citer — wired by a control loop. The loop's defining feature is the **back-edge**
(*evaluator → reformulate*): when results aren't good enough, the agent rewrites
the query and searches again. That back-edge is what makes it an *agent* rather
than a fixed pipeline (DAG).

```bash
pip install raglab
```

> **Fresh start (v0.2.0+).** This repo took over the `raglab` PyPI name. The
> earlier backend (PyPI 0.0.x–0.1.x) was renamed and now lives at
> [addaix/raglab_bak](https://github.com/addaix/raglab_bak) (PyPI:
> [`raglab_bak`](https://pypi.org/project/raglab_bak/)). The API below is the new
> agent layer.

---

## Quick start

The default agent runs **fully offline** — no LLM, no network. Point it at one or
more `ir` corpora and call it like a function; you get back ranked
[`ir.SearchHit`](https://github.com/i2mint/ir)s.

```python
import ir
import raglab

# Register named ir corpora as the agent's sources, then search across them.
sources = raglab.ir_sources("skills", "reports", mode="hybrid")
agent = raglab.make_search_agent(sources)
results = agent("how do I deploy the app")  # -> list[ir.SearchHit], ranked
```

A self-contained, runnable example (build a tiny corpus, search it):

```python
import ir
from ir.store import CorpusStore
import raglab

docs = {
    "deploy": "deploy the app to the server with systemd units",
    "embed": "embed text with a model and cache the vectors",
    "search": "vector similarity search with metadata filters",
}
corpus = ir.build(
    ir.CorpusSource.from_mapping(docs, name="t", strategy=ir.WholeText()),
    store=CorpusStore.memory(),
    embedder="light",  # numpy-only, no model download
)

agent = raglab.make_search_agent({"t": ir.as_retriever(corpus, k=3)})
results = agent("deploy the app to the server")
results[0].artifact_id  # -> "deploy"
```

`sources` is any `Mapping[name, Retriever]`. Build it with
[`raglab.ir_sources(*names, **defaults)`](#sources), with `ir.retrievers()` (a
lazy live view over everything ir has registered), or by hand:
`{"skills": ir.as_retriever("skills")}`.

---

## The mental model

```
           ┌─────────── back-edge (re-query when insufficient) ──────────┐
           ▼                                                             │
  Query → Planner → [SubTask] → Formulator → [LowLevelQuery] → Retriever(s)
                                                                   │
                                          accumulated hits ◄───────┘
                                                   │
                                            Evaluator ──► Judgement(relevant, sufficient, refinement)
                                                   │              └─ refinement ⟲ back-edge
                                          (sufficient / budget hit)
                                                   ▼
                                       Reranker (fan-in) → Citer → ranked results
```

The **loop is fixed**; every *decision* in it is an injected **role** (a plain
callable satisfying a `Protocol`). v1 ships every role with a sensible default,
so the simple path — `make_search_agent(sources)("query")` — just works. You turn
capabilities on by swapping in richer roles.

| Role | Decides | Default (no-LLM) | Inject an LLM version to… |
|------|---------|------------------|---------------------------|
| **Planner** | decompose query → sub-tasks, pick sources | `single_subtask_planner` (one task, all sources) | split a query into parallel sub-goals |
| **Formulator** | sub-task → concrete search queries | `identity_formulator` (verbatim) | rewrite / expand / HyDE (`make_llm_formulator`) |
| **Retriever** | run one query against one source | *injected at the leaf* (`ir.as_retriever`) | add web / SQL / graph backends |
| **Evaluator** | relevance + sufficiency + **back-edge** | `passthrough_evaluator` (keep all, stop) | judge sufficiency & re-query (`make_llm_evaluator`) |
| **Reranker** | final cross-source ordering (fan-in) | `rrf_reranker` (rank fusion) | weight sources, change fusion |
| **Citer** | confirm each result supports its use | `identity_citer` (no-op) | *(generation/verification lives in `srag`)* |

**`Budget`** bounds the loop (`max_rounds`, `max_sources_per_task`,
`max_results_per_task`) — the safety net under the (harder) sufficiency decision.

---

## Turning on query understanding and the back-edge

Inject the two LLM roles to upgrade the offline slice into a real agent. Both
build their model lazily on [`oa`](https://github.com/thorwhalen/oa) **only when
invoked**, so `import raglab` stays offline.

```python
import raglab

agent = raglab.make_search_agent(
    sources,
    formulator=raglab.make_llm_formulator(),  # rewrite/expand the query (HyDE)
    evaluator=raglab.make_llm_evaluator(),  # sufficiency + the back-edge
)
results = agent("how do I deploy the app")
```

How the Evaluator splits the decision (a load-bearing boundary):

- **Relevance is `ir`'s.** Each round the accumulated hits pass through
  `ir.select` (default strategy `"conservative"` — distractor-robust); the
  committed subset becomes `Judgement.relevant`. LLM relevance scoring is
  known-fragile, so the LLM stays out of it.
- **Sufficiency is the LLM's.** Informed by ir's model-free
  `Selection.sufficient` hint, the judge decides whether the committed subset
  actually satisfies the goal. If not, it emits an improved query — a
  `refinement` `SubTask` over the same sources. **That refinement is the
  back-edge.**

Both builders accept an injectable double (`formulate=` / `judge=`) for
deterministic tests, and both **fail safe**: a formulator error falls back to the
raw query (never makes retrieval worse); a judge error returns no refinement
(the loop's break condition — a failing judge can never spin forever).

---

## Multi-source search and the fan-in reranker

The agent is **multi-source by default**. The loop stamps each hit's provenance
(`hit.source`), and the default fan-in reranker, `rrf_reranker`, merges
heterogeneous sources **by rank, never by raw score**.

This matters because scores from different corpora / embedders / retrieval modes
are *incommensurable* — a cosine score (~0–1) and a BM25 score (~0–50) cannot be
compared directly. Reciprocal Rank Fusion (RRF) sidesteps this: within one source
raw scores order and dedup that source's hits (one scale, sound); across sources
only **ranks** interact.

```python
# Two sources on different score scales:
#   rrf_reranker (default): interleaves both sources' rank-1 hits first.
#   score_reranker (opt-in): a raw-score sort — only sound when every source
#                            shares one scale (same embedder + mode).
agent = raglab.make_search_agent(sources, reranker=raglab.score_reranker)
```

Colliding `artifact_id`s from different sources stay distinct (identity is
`(source, artifact_id)`). Each fused hit keeps its pre-fusion magnitude in
`metadata["source_score"]`. For per-source trust weights or a custom `rrf_k`:

```python
reranker = raglab.make_rrf_reranker(weights={"skills": 2.0, "web": 0.5}, rrf_k=60)
agent = raglab.make_search_agent(sources, reranker=reranker)
```

---

## Customizing roles

Every role is just a callable matching a `Protocol`, so you can inject your own:

```python
from raglab import SubTask, LowLevelQuery, Judgement, make_search_agent


def my_planner(query, sources):
    # split into two sub-goals over all sources
    return [
        SubTask(goal=g, sources=tuple(sources))
        for g in (query.text, query.text + " examples")
    ]


def my_formulator(task, source):
    return [LowLevelQuery(source=source, query=task.goal, params={"k": 10})]


agent = make_search_agent(sources, planner=my_planner, formulator=my_formulator)
```

A **custom Retriever** is any `callable(query, **params) -> Sequence[ir.SearchHit]`.
It must return `ir.SearchHit` instances (the `Result` alias) — the loop stamps
provenance on its output, so duck-typed hit objects raise at the tagging step.
ir-backed retrievers self-attribute their corpus name.

---

## API at a glance

Everything is exported from the top-level `raglab` namespace.

**Value types** (frozen dataclasses, plain data):

| Type | Fields |
|------|--------|
| `Query` | `text`, `constraints` |
| `SubTask` | `goal`, `sources: tuple[str, ...]` |
| `LowLevelQuery` | `source`, `query`, `params` (per-call retriever overrides) |
| `Judgement` | `relevant`, `sufficient`, `refinement: SubTask \| None` |
| `Budget` | `max_rounds=3`, `max_sources_per_task=4`, `max_results_per_task=50` |
| `Result` | alias for `ir.SearchHit` |

**Role Protocols** (the open-closed strategy seams): `Planner`, `Formulator`,
`Retriever` (re-exported from `ir`), `Evaluator`, `Reranker`, `Citer`.

**Orchestrator & builders:**

| Name | What it does |
|------|--------------|
| `make_search_agent(sources, *, planner, formulator, evaluator, reranker, citer, budget)` | Build a `SingleContextAgent` with smart defaults for every role. |
| `SingleContextAgent` | The orchestrator dataclass: one ReAct-style loop, sequential sub-tasks. |
| <a name="sources"></a>`ir_sources(*names, **search_defaults)` | A `{name: Retriever}` registry backed by named ir corpora. |

**Default roles (the no-LLM thin slice):** `single_subtask_planner`,
`identity_formulator`, `passthrough_evaluator`, `score_reranker`, `rrf_reranker`,
`make_rrf_reranker(*, rrf_k, weights, identity)`, `identity_citer`.

**LLM roles:** `make_llm_formulator(*, formulate=None, params=None, **make_kwargs)`,
`make_llm_evaluator(*, judge=None, select_strategy="conservative", select_kwargs=None, prerank=None, prompt=EVALUATION_PROMPT, ...)`,
and the default `EVALUATION_PROMPT` string.

---

## Architecture & boundaries

`raglab` **owns the agent**: the value types, the role Protocols, the control
loop with the back-edge, the budget governor, the source registry, and the
cross-source fan-in.

It **consumes from `ir`** (and does not reimplement): `ir.as_retriever` (corpus →
Retriever), `ir.retrievers()` (the live registry view), `ir.make_llm_formulator`
(the query rewriter), `ir.select` + its `Selection.sufficient` signal (relevance
+ the sufficiency hint), and `ir.fuse_hits` (RRF). The `Result` type and the
`Retriever` contract are **ir's** (single source of truth).

**Dependency direction is one-way:** `raglab` imports `ir` (and `oa` for the LLM
roles, lazily); **`ir` never imports `raglab`.** Keeping `oa` opt-in is what lets
`import raglab` stay offline.

What lives **elsewhere:** answer synthesis / generation and the
Citer/Verifier (which needs a *generated claim* to verify) belong in the
RAG/generation layer (`srag`), not the search agent. raglab's deliverable is
**pointers + extractions**, not an essay.

Where it sits in the ecosystem: **`raglab` → `ir` → {`ef`, `vd`}** — `ir` is the
retrieval substrate, `ef` owns embedders/segmenters, `vd` is the vector-store
facade.

---

## Status & roadmap

Shipped at **0.2.4**: the `SingleContextAgent` control loop with the back-edge,
all six role Protocols, the no-LLM defaults, the LLM Formulator + Evaluator, and
the RRF cross-source fan-in reranker. The roles, value types, and loop are
stable; richer strategies are added behind the same contracts.

The living roadmap is **[issue #2](https://github.com/addaix/raglab/issues/2)**
(Composable Search Agent on ir, the *ir_09* reference design). Not yet built:

- **Budget governor refinements** — the per-task results cap currently truncates
  in arrival order rather than by rank
  ([#5](https://github.com/addaix/raglab/issues/5)); termination as a
  separately injected policy is planned.
- **PurposeStore** — persistent, purpose-indexed memory overlay
  ([#6](https://github.com/addaix/raglab/issues/6)). There is no cross-run
  memory today.
- **`MultiAgentAgent`** — one subagent per sub-task/source (breadth-first, ~15×
  cost); promotion swaps only the orchestrator, keeping role contracts identical.

---

## House style

Functional over OOP; SOLID when OOP; facades, SSOT, dependency injection;
progressive disclosure; keyword-only args beyond the third positional;
`collections.abc` + frozen `dataclass`es; `Protocol`s for the role seams. Tests
are hermetic (fake retrievers, injected LLM doubles) with a few end-to-end checks
over a real `ir` corpus using the light, numpy-only embedder.
