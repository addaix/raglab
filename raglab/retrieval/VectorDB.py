"""VectorDB"""

# VectorDB class
from typing import Mapping, Iterable, Dict, Any
from dataclasses import dataclass, field
from types import SimpleNamespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import cached_property

DocKey = str
DocValue = str


from typing import Tuple, List, Mapping

SegmentKey = Tuple[DocKey, int, int]


def generate_split_keys(
    docs: dict, text_splitter: RecursiveCharacterTextSplitter, metadatas: List[dict]
) -> List[Tuple[str, int, int]]:

    documents = text_splitter.create_documents(
        list(docs.values()), metadatas=list({"document_name": k} for k in docs.keys())
    )

    return [
        (
            d.metadata["document_name"],
            d.metadata["start_index"],
            d.metadata["start_index"] + len(d.page_content),
        )
        for d in documents
    ]


class SegmentMapping:
    """A class to represent a mapping between segments and documents."""

    def __init__(self, docs: Mapping, segment_keys: List[SegmentKey]):
        self.docs = docs
        self.segment_keys = segment_keys
        self.document_keys = list(docs.keys())

    def __iter__(self):
        yield from self.segment_keys

    def __getitem__(self, key: SegmentKey):
        if isinstance(key, str):
            return self.docs[key]
        elif isinstance(key, Tuple):
            doc_key, start_idx, end_idx = key
            return self.docs[doc_key][start_idx:end_idx]
        else:
            raise TypeError("Key must be a string or a tuple")

    def __setitem__(self, key: SegmentKey, value: str):
        if isinstance(key, str):
            self.docs[key] = value
            return
        else:
            doc_key, start_idx, end_idx = key
            self.segment_keys.append(key)
            self.docs[doc_key] = (
                self.docs.get(doc_key, "")[:start_idx]
                + value
                + self.docs.get(doc_key, "")[end_idx:]
            )

    def __add__(self, other):
        """Add two SegmentMapping objects together. This will concatenate the documents and segment keys."""
        return SegmentMapping(
            {**self.docs, **other.docs}, self.segment_keys + other.segment_keys
        )

    def __len__(self):
        return len(self.segment_keys)

    def __contains__(self, key: SegmentKey):
        if isinstance(key, str):
            return key in self.document_keys
        elif isinstance(key, Tuple):
            return key in self.segment_keys
        else:
            raise TypeError("Key must be a string or a tuple")

    def __repr__(self):
        representation = ""
        for key in self.segment_keys:
            representation += str(key) + " : " + str(self.__getitem__(key)) + "\n"
        return representation

    def values(self):
        for key in self.segment_keys:
            yield self.__getitem__(key)


@dataclass
class ChunkDB:
    """Contains a mapping from document keys to document content. Keys can either be a tuple (doc_key, start_idx, end_idx) or a string doc_key.
    The documents are split into segments and the segments are stored in a SegmentMapping object.
    The SegmentMapping object is a mapping from segment keys to segments."""

    docs: Mapping[DocKey, DocValue]
    # kwargs: Dict[str, Any] = field(default_factory=dict)
    chunk_size: int = 400
    chunk_overlap: int = 100

    @cached_property
    def text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=min(self.chunk_overlap, self.chunk_size - 1),
            length_function=len,
            add_start_index=True,  # enforce the start index
            is_separator_regex=False,
        )

    @cached_property
    def segments(self):
        return self.mk_segment_store(self.docs)

    @cached_property
    def segment_store(self):
        return SimpleNamespace(segment_keys=self.segments)

    @cached_property
    def document_store(self):
        return SimpleNamespace(docs=self.docs)

    @cached_property
    def mall(self):
        return SimpleNamespace(docs=self.docs, segment_keys=self.segments)

    def add(self, new_documents: Mapping[DocKey, DocValue], when_exists: str = "raise"):
        """Add new documents to the database. When_exists can be "raise", "skip" or "overwrite.
        New documents has to be a mapping with the document key as the key and the document content as the value.
        """

        for new_doc_key in new_documents:
            if new_doc_key in self.segments:
                if when_exists == "skip":
                    continue
                elif when_exists == "overwrite":
                    self.segments[new_doc_key] = new_documents[new_doc_key]
                else:
                    raise ValueError(
                        f"Document with key {new_doc_key} already exists. Use 'when_exists' to handle this case."
                    )
            else:
                self.segments = self.segments + self.mk_segment_store(new_documents)

    def mk_segment_store(self, docs):
        """Creates and returns a segment store from a dictionary of documents. Segment store is a mapping from segment keys to segments."""
        segment_keys = generate_split_keys(docs, self.text_splitter, metadatas=[])
        return SegmentMapping(docs, segment_keys)

    def search(self, query: str, k: int = 1) -> Iterable[DocKey]:
        # TODO : return (yields) the k most similar documents to the query
        pass

    def add_metadata(self, metadata: Mapping[DocKey, dict]):
        # TODO : add metadata to the documents
        pass
