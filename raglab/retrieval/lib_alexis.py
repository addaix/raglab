""" This module provides functions to retrieve documents from a corpus based on a query. """

from config2py import config_getter, get_app_data_folder, process_path
from dotenv import load_dotenv
from dol import wrap_kvs, Pipe
from functools import partial
from heapq import nlargest
from importlib.resources import files
from io import BytesIO
from i2 import Namespace

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from meshed import DAG
from msword import bytes_to_doc, get_text_from_docx
import numpy as np
import oa
import os
from docx import Document
from docx2python import docx2python
from docx2python.iterators import iter_paragraphs
import json
import jsonschema
from jsonschema import validate
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import pickle
import tiktoken
from typing import Mapping, List, Optional, Any, Tuple, Callable
import PyPDF2

DocKey = str

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
chat = ChatOpenAI(
    temperature=0, model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY")  # type: ignore
).predict

docs = {
    "doc1": "John and Mary went to the park",
    "doc2": "Mary and John went to the cinema",
}


# def generate_split_keys(
#     docs: Mapping[DocKey, str],
#     chunk_size: int,
#     chunk_overlap: int,
#     separators: Optional[List[str]] = None,
#     keep_separator: bool = True,
#     is_separator_regex: bool = False,
#     **kwargs: Any,
# ) -> List[Tuple[str, int, int]]:
#     """Generate the split keys for the documents: Split key is a tuple of (document_name, start_index, end_index)"""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         add_start_index=True,  # enforce the start index
#         is_separator_regex=is_separator_regex,
#         keep_separator=keep_separator,
#     )
#     documents = text_splitter.create_documents(
#         list(docs.values()), metadatas=list({"document_name": k} for k in docs.keys())
#     )

#     return [
#         (
#             d.metadata["document_name"],
#             d.metadata["start_index"],
#             d.metadata["start_index"] + len(d.page_content),
#         )
#         for d in documents
#     ]


def split_text(text, max_chunk_size, separators=None):
    if separators is None:
        separators = ["\n", ".", "!", "?"]

    def recursive_split(text, max_chunk_size, separators):
        if len(text) <= max_chunk_size:
            return [(0, len(text))]

        best_split_index = -1
        for sep in separators:
            split_index = text.rfind(sep, 0, max_chunk_size)
            if split_index != -1:
                best_split_index = split_index + len(sep)
                break

        if best_split_index == -1:
            best_split_index = max_chunk_size

        left_splits = recursive_split(
            text[:best_split_index], max_chunk_size, separators
        )
        right_splits = recursive_split(
            text[best_split_index:], max_chunk_size, separators
        )
        right_splits = [
            (start + best_split_index, end + best_split_index)
            for start, end in right_splits
        ]

        return left_splits + right_splits

    indices = recursive_split(text, max_chunk_size, separators)
    return indices


def generate_split_keys(
    docs: Mapping[DocKey, str],
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
) -> List[Tuple[str, int, int]]:
    segment_keys = []
    for doc_key, doc_text in docs.items():
        split_indices = split_text(doc_text, chunk_size, separators)
        for i, (start, end) in enumerate(split_indices):
            segment_keys.append(
                (doc_key, max(0, start - chunk_overlap), end + chunk_overlap)
            )
    return segment_keys


def query_embedding(query: str) -> np.ndarray:
    return np.array(embeddings_model.embed_query(query))


def chunker(chunk_overlap: int, **kwargs):
    return partial(generate_split_keys, chunk_overlap=chunk_overlap)


_generate_split_keys = partial(generate_split_keys, chunk_overlap=100)

chunker_type = Callable[[Mapping[DocKey, str]], List[Tuple[str, int, int]]]


# TODO : @cache_result : cache the result of this function
def segment_keys(
    documents: Mapping[DocKey, str],
    chunker: Callable[
        [Mapping[DocKey, str], int], List[Tuple[str, int, int]]
    ] = _generate_split_keys,
    max_chunk_size: int = 1000,
) -> List[Tuple[str, int, int]]:
    return chunker(documents, max_chunk_size)


def doc_embeddings(
    documents: Mapping[DocKey, str],
    segment_keys: List[Tuple[str, int, int]],
    embedding_function: Callable[
        [List[str]], List[np.ndarray]
    ] = embeddings_model.embed_documents,
) -> Mapping[Tuple[str, int, int], np.ndarray]:
    return dict(
        zip(
            segment_keys,
            np.array(
                embedding_function([documents[s[0]][s[1] : s[2]] for s in segment_keys])
            ),
        )
    )


def dump_embeddings(
    doc_embeddings: Mapping[Tuple[str, int, int], np.ndarray], path: str
):
    pass


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


def query(user_query: str, process_function: Callable[[str], str] = lambda x: x) -> str:
    return process_function(user_query)


def query_answer(
    query: str,
    documents: Mapping[DocKey, str],
    top_k_segments: List[Tuple[str, int, int]],
    prompt_template: str = "Answer question {query} using the following documents: {documents}",
    chat_model=chat,
) -> str:
    aggregated_text = ""
    for segment_key in top_k_segments:
        aggregated_text += documents[segment_key[0]][segment_key[1] : segment_key[2]]
    return oa.prompt_function(prompt_template, prompt_func=chat_model)(
        query=query,
        documents=aggregated_text,
    )


funcs = [
    segment_keys,
    doc_embeddings,
    top_k_segments,
    # chunker,
    query_embedding,
    query_answer,
    query,
]
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


def tokens(string: str, encoding_name: str = "cl100k_base") -> List[str]:
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(string)


# def read_pdf_text(pdf_reader):
#     text_pages = []
#     for page in pdf_reader.pages:
#         text_pages.append(page.extract_text())
#     return text_pages


# def read_pdf(file, *, page_sep="\n--------------\n") -> str:
#     with pdfplumber.open(file) as pdf:
#         return page_sep.join(read_pdf_text(pdf))


def read_pdf_text_with_plumber(pdf_reader):
    text_pages = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return text_pages


def read_pdf_text_with_pypdf2(pdf_reader):
    text_pages = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    return text_pages


def read_pdf(file, *, page_sep="\n--------------\n") -> str:
    try:
        with pdfplumber.open(BytesIO(file)) as pdf:
            text_pages = read_pdf_text_with_plumber(pdf)
            combined_text = page_sep.join(text_pages)
            if combined_text.strip():  # Check if the extracted text is not empty
                return combined_text
            else:
                raise ValueError("Empty text extracted with pdfplumber")
    except Exception as e:
        print(f"pdfplumber failed with error: {e}, trying PyPDF2...")
        try:
            bio = BytesIO(file)  # Ensure BytesIO object is read correctly
            reader = PyPDF2.PdfReader(bio)
            text_pages = read_pdf_text_with_pypdf2(reader)
            combined_text = page_sep.join(text_pages)
            if combined_text.strip():  # Check if the extracted text is not empty
                return combined_text
            else:
                raise ValueError("Empty text extracted with PyPDF2")
        except Exception as e2:
            print(f"PyPDF2 also failed with error: {e2}")
            raise e2  # Re-raise the last exception


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
    ".pdf": read_pdf,
    ".docx": Pipe(BytesIO, full_docx_decoder),
    ".pkl": lambda obj: pickle.loads(obj),
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


def extension_based_encoding(k, v):
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
    source[key] = extension_based_encoding(key, value)
    return value


def simple_language_detection(text: str):
    """Detects between english and french regading the text language."""
    french_words = set(
        [
            "du",
            "de",
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "au",
            "aux",
            "en",
            "et",
            "ou",
            "qui",
            "que",
        ]
    )
    english_words = set(
        [
            "the",
            "a",
            "an",
            "of",
            "to",
            "and",
            "or",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
        ]
    )
    text = text.lower()
    french_count = sum([1 for word in text.split() if word in french_words])
    english_count = sum([1 for word in text.split() if word in english_words])
    if french_count == english_count:
        return None
    if french_count == english_count == 0:
        return None
    if french_count > english_count:
        return "french"
    else:
        return "english"


def populate_local_user_folders(defaults, local_user_folders):
    for k in defaults:
        if k not in local_user_folders:
            local_user_folders[k] = defaults[k]
    return defaults


def validate_json(data, schema):
    try:
        validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        print(f"Validation error: {err.message}")
        return False
    return True
