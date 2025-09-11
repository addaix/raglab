"""Component registry management for vd_facade_1"""

from typing import Callable, Any, Optional, MutableMapping
from functools import partial

# Component registries - global dictionaries for each component type
segmenters: MutableMapping[str, Callable] = {}
embedders: MutableMapping[str, Callable] = {}
vector_stores: MutableMapping[str, Callable[..., Any]] = {}  # Factories
indexers: MutableMapping[str, Callable[..., Any]] = {}  # Factories


def register_component(registry: MutableMapping, name: Optional[str] = None):
    """Generic component registration decorator"""

    def decorator(component: Callable):
        key = name or getattr(component, "__name__", None) or repr(component)
        registry[key] = component
        return component

    return decorator


# Specific registration decorators
register_segmenter = partial(register_component, segmenters)
register_embedder = partial(register_component, embedders)
register_vector_store = partial(register_component, vector_stores)
register_indexer = partial(register_component, indexers)


def get_component(registry: MutableMapping, spec: Any, **kwargs) -> Any:
    """
    Retrieve a component from registry by specification.

    spec may be:
      - callable: returned or instantiated with kwargs
      - str: key into registry
      - dict: {name: params} where params merged with kwargs
    """
    if callable(spec) and not isinstance(spec, str):
        try:
            return spec(**kwargs) if kwargs else spec
        except TypeError:
            # If callable doesn't accept kwargs, return it directly
            return spec

    if isinstance(spec, str):
        if spec not in registry:
            raise KeyError(f"Component '{spec}' not found in registry")
        component = registry[spec]
        # If the registered component is a class, instantiate it so callers
        # get an instance (useful for embedder classes that provide __call__).
        if isinstance(component, type):
            return component(**kwargs) if kwargs else component()
        return component(**kwargs) if kwargs else component

    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError("Component spec dict must have exactly one key")
        name, params = next(iter(spec.items()))
        if name not in registry:
            raise KeyError(f"Component '{name}' not found in registry")
        component = registry[name]
        merged = {**params, **kwargs}
        # Instantiate classes when specified by name; otherwise call functions
        if isinstance(component, type):
            return component(**merged) if merged else component()
        return component(**merged) if merged else component

    raise TypeError(f"Invalid component spec type: {type(spec)}")
