"""Configuration for vd_facade_1"""

import os
from dataclasses import dataclass

ENV_PREFIX = "VD_FACADE_"


@dataclass
class FacadeConfig:
    default_segmenter: str = "simple_sentence"
    default_embedder: str = "simple_count"
    default_vector_store: str = "memory"
    auto_segment: bool = True
    auto_embed: bool = True

    @classmethod
    def from_env(cls):
        kwargs = {}
        for field in cls.__dataclass_fields__:
            env = f"{ENV_PREFIX}{field.upper()}"
            if env in os.environ:
                val = os.environ[env]
                t = cls.__dataclass_fields__[field].type
                if t is bool:
                    val = val.lower() in ("1", "true", "yes")
                kwargs[field] = val
        return cls(**kwargs)


config = FacadeConfig.from_env()
