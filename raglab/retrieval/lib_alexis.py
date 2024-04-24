""" This module provides functions to retrieve documents from a corpus based on a query. """

from typing import Mapping, List, Tuple, Callable
from heapq import nlargest
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from config2py import config_getter
import numpy as np
from meshed import DAG

DocKey = str

OPENAI_API_KEY = config_getter("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


docs = {
    "doc1": "John and Mary went to the park",
    "doc2": "Mary and John went to the cinema",
}


def generate_split_keys(
    docs: Mapping[DocKey, str], chunk_size
) -> List[Tuple[str, int, int]]:
    """Generate the split keys for the documents: Split key is a tuple of (document_name, start_index, end_index)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,  # enforce the start index
        is_separator_regex=False,
    )
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


def query_embedding(query: str) -> np.ndarray:
    return np.array(embeddings_model.embed_query(query))


def segment_keys(
    documents: Mapping[DocKey, str], chunk_size: int = 30
) -> List[Tuple[str, int, int]]:
    return generate_split_keys(documents, chunk_size)


def doc_embeddings(
    documents: Mapping[DocKey, str], segment_keys: List[Tuple[str, int, int]]
) -> Mapping[Tuple[str, int, int], np.ndarray]:
    return dict(
        zip(
            segment_keys,
            np.array(
                embeddings_model.embed_documents(
                    [documents[doc_key] for doc_key in documents]
                )
            ),
        )
    )


def top_k_segments(
    query_embedding: np.ndarray,
    doc_embeddings: Mapping[Tuple[str, int, int], np.ndarray],
    k: int = 1,
    distance_metric: Callable[[np.ndarray, np.ndarray], float] = cosine_similarity,
) -> List[Tuple[str, int, int]]:
    top_k = nlargest(
        k,
        (
            (
                distance_metric(
                    query_embedding.reshape(1, -1), doc_embedding.reshape(1, -1)
                ),
                key,
            )
            for key, doc_embedding in doc_embeddings.items()
        ),
    )
    return [key for _, key in top_k]


funcs = [segment_keys, doc_embeddings, top_k_segments, query_embedding]
dag = DAG(funcs)

dag.dot_digraph()


def return_save_bytes(save_function):
    """Return bytes from a save function.

    :param save_function: A function that saves to a file-like object
    :return: The serialization bytes

    """
    import io

    io_target = io.BytesIO()
    with io_target as f:
        save_function(f)
        io_target.seek(0)
        return io_target.read()


def num_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def read_pdf_text(pdf_reader):
    text_pages = []
    for page in pdf_reader.pages:
        text_pages.append(page.extract_text())
    return text_pages


def read_pdf(file, *, page_sep="\n--------------\n") -> str:
    with pdfplumber.open(file) as pdf:
        return page_sep.join(read_pdf_text(pdf))


def full_docx_decoder(doc_bytes):
    text = get_text_from_docx(Document(doc_bytes))
    doc = docx2python(doc_bytes)
    added_header = "\n\n".join(iter_paragraphs(doc.header)) + text
    added_footer = added_header + "\n\n".join(iter_paragraphs(doc.footer))
    return added_footer


# Map file extensions to decoding functions
extension_to_decoder = {
    ".txt": lambda obj: obj.decode("utf-8"),
    ".json": json.loads,
    ".pdf": Pipe(BytesIO, read_pdf),
    ".docx": Pipe(BytesIO, full_docx_decoder),
}

extension_to_encoder = {
    ".txt": lambda obj: obj.encode("utf-8"),
    ".json": json.dumps,
    ".pdf": lambda obj: obj,
    ".docx": lambda obj: obj,
}


def extension_based_decoding(k, v):
    ext = "." + k.split(".")[-1]
    decoder = extension_to_decoder.get(ext, None)
    if decoder is None:
        decoder = extension_to_decoder[".txt"]
    return decoder(v)


def extension_base_encoding(k, v):
    ext = "." + k.split(".")[-1]
    encoder = extension_to_encoder.get(ext, None)
    if encoder is None:
        encoder = extension_to_encoder[".txt"]
    return encoder(v)


def extension_base_wrap(store):
    return wrap_kvs(
        store, postget=extension_based_decoding
    )  # , preset=extension_base_encoding)


from typing import List


def get_config(key, sources) -> str:
    """if not key in source ask user and put it in the source"""
    for source in sources:
        if key in source:
            return source[key]
        else:
            continue

    value = input(f"Please enter the value for {key} and press enter")
    source[key] = extension_base_encoding(key, value)
    return value
