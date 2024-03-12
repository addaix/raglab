# raglab

Backend of a system to explore RAG operations.

To install:	```pip install raglab```

**IMPORTANT NOTE: This is work-in-progress and shouldn't be used, yet.**

`raglab` is meant to be a medley of interoperable functionalities around retrieval 
(or "search") and generation (with LLMs).
It's architecture is structured so that operations can be done in python, 
at the command line, through a web service (or any OpenAPI language binders) or from a 
GUI, where each layer's interface is mapped from the other in a consistent 
(and usually automatic) manner.

# Work-in-progress notes

## Simple resources CRUD

Some functionalities/operations can simply be described as 
(1) a set of operations 
(2) over a kind of data

We'll sometimes lazily say CRUD for the "set of operations", but we don't always mean Create-Read-Update-Delete. In fact, most of the time we don't mean those four operations. Instead, since python is our backend language and we [aim at using base interfaces](https://medium.com/@thorwhalen1/mastering-modular-design-the-art-of-using-base-interface-facades-9d4a08d550d1) as much as possible, our "set of operations" will often be inspired from python's [Collections Abstract Base Classes](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes). 

So many concrete use cases of this abstraction are parametrized by
* the set of operation (example, read, write, list, delete...)
* a name designating the kind/type of data
* optionally a schema that describes what the structure and contents of that kind of data must be

Further, by making the interface consistent (for example, and preferably, with the interfaces of  [Collections Abstract Base Classes](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes)), it makes it possible to automate how the python interface maps to a REST API interface (see 
[Mapping python collections builtin methods to REST API patterns](https://github.com/i2mint/py2http/discussions/9))
and/or a UI interface (see [Collections Operations UI Design Patterns](https://github.com/i2mint/oui/discussions/14)).

### Examples

* `permissions` (keyed by users, groups of users, projects etc.)
* `datasets` (again keyed by users etc.)
* `prompt_templates`
* `embedders`
* `dataset_query_results`
* `query_results_aggregators`
* `query_results_aggregates`
* `tag`: For example `tag, dataset`, `tag, group_of_datasets`, `tag, anything_really`. Note that tagging, or annotating can be applied to any kind of data, and is a dual of "grouping", and subsumes hierarchical organization. See  [Persisting groups of items (tagging stuff)](https://github.com/i2mint/i2mint/blob/main/misc/Persisting%20groups%20of%20items%20(tagging%20stuff).ipynb) for designs regarding this.

