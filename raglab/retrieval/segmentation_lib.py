""" This module contains functions to segment a text into chunks of a maximum number of tokens. """

from config2py import config_getter
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from langchain_openai import OpenAIEmbeddings
from config2py import config_getter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from meshed import DAG, code_to_dag
import matplotlib.pyplot as plt
from raglab.retrieval.lib_alexis import num_tokens

MAX_TOKENS = 8_000
OPENAI_API_KEY = config_getter("OPENAI_API_KEY")

# # Download the necessary NLTK data files (only needed once)
# nltk.download("punkt")
# # Segment the text into sentences
# sentence_splits = sent_tokenize
# sentence_splits.__name__ = "sentence_splits"


def sentence_splits_ids(text, sep=[".", "!", "?"]):
    """Returns the indices of the sentence splits made from characters in sep."""
    indices = []
    start = 0
    for i, char in enumerate(text):
        if char in sep:
            indices.append((start, i + 1))
            start = i + 1
    if start < len(text):
        indices.append((start, len(text)))
    return indices


def is_numerical(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_meaningful(sentence):
    """FIlters sentences that are not meaningful. It removes stopwords and punctuation."""
    stop_words = set(stopwords.words("english")).union(set(stopwords.words("french")))
    words = word_tokenize(sentence)
    # Filter out punctuation and stopwords
    meaningful_words = [
        word
        for word in words
        if word.lower() not in stop_words
        and word not in punctuation
        and len(word) > 1
        and not is_numerical(word)
    ]
    return len(meaningful_words) > 0


def filtered_sentence_split_ids(sentence_splits_ids, text):
    """Filters out the sentence splits that are not meaningful."""
    return [
        (start, stop)
        for start, stop in sentence_splits_ids
        if is_meaningful(text[start:stop])
    ]


def sentence_splits(text, filtered_sentence_split_ids):
    return [text[start:stop] for start, stop in filtered_sentence_split_ids]


# def meaningful_sentences(sentence_splits):
#     return [sentence for sentence in sentence_splits if is_meaningful(sentence)]


def sentence_embeddings(sentence_splits, api_key=OPENAI_API_KEY, dimension=512):
    """Returns the embeddings of the sentences using the OpenAI API."""
    embeddings_model = OpenAIEmbeddings(
        api_key=api_key, model="text-embedding-3-small", dimensions=dimension
    )
    return np.array(embeddings_model.embed_documents(sentence_splits))


def consecutive_cosines(sentence_embeddings):
    """Returns the cosine similarity between consecutive sentences. The last cosine is not computed since there is no next sentence."""
    z = zip(sentence_embeddings, sentence_embeddings[1:])
    cosines = [cosine_similarity([a], [b])[0][0] for a, b in z]
    return cosines


def gap_sentence(consecutive_cosines):
    """
    Returns the index of the sentence that is the most different from the next sentence.
    """
    if len(consecutive_cosines) < 3:
        return 0
    sentence_cut_id = np.argmin(consecutive_cosines[:-1])
    return sentence_cut_id


def gap_character(gap_sentence, sentence_split_ids):
    return sentence_split_ids[gap_sentence][1] + 1


def sentence_num_tokens(sentence_splits):
    return [num_tokens(sentence) for sentence in sentence_splits]


def max_sentence_tokens(sentence_num_tokens):
    return max(sentence_num_tokens)


def segment_keys(
    sentence_num_tokens,
    consecutive_cosines,
    filtered_sentence_split_ids,
    max_tokens=MAX_TOKENS,
    start=0,
):
    """Returns the indices of the sentence splits that segment the text into chunks of a maximum number of tokens."""
    min_tokens = max_sentence_tokens(sentence_num_tokens)  # TODO as parameter
    assert max_tokens > min_tokens, "max_tokens must be greater than min_tokens"

    if sum(sentence_num_tokens) <= max_tokens or len(sentence_num_tokens) == 1:
        return [(start, filtered_sentence_split_ids[-1][1])]

    gap_sentence_id = gap_sentence(consecutive_cosines) + 1
    left_cosines = consecutive_cosines[:gap_sentence_id]
    left_split_ids = filtered_sentence_split_ids[:gap_sentence_id]
    left_num_tokens = sentence_num_tokens[:gap_sentence_id]
    right_cosines = consecutive_cosines[gap_sentence_id:]
    right_split_ids = filtered_sentence_split_ids[gap_sentence_id:]
    right_num_tokens = sentence_num_tokens[gap_sentence_id:]

    return segment_keys(
        left_num_tokens,
        left_cosines,
        left_split_ids,
        max_tokens=max_tokens,
        start=start,
    ) + segment_keys(
        right_num_tokens,
        right_cosines,
        right_split_ids,
        max_tokens=max_tokens,
        start=right_split_ids[0][0],
    )


def sentence_cut_ids(segment_keys, filtered_sentence_split_ids):
    """returns the indices of the sentences to cut. The cut sentence belong to the previous segment."""
    stop_chars = [stop for start, stop in segment_keys[:-1]]
    # get corresponding sentence id
    sentence_cut_ids = [
        i
        for i, (start, stop) in enumerate(filtered_sentence_split_ids)
        if stop in stop_chars
    ]
    return sentence_cut_ids


def text_segments(text, segment_keys):
    return [text[start:stop] for start, stop in segment_keys]


def chunk_text(character_chunk_ids, text):
    return [text[start:stop] for start, stop in character_chunk_ids]


def display_cut_ids(
    sentence_splits,
    sentence_embeddings,
    consecutive_cosines,
    sentence_cut_ids,
    verbose=True,
):
    num_sentences = len(sentence_splits)
    if not verbose:
        return
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(consecutive_cosines)
    plt.scatter(
        [s for s in sentence_cut_ids],
        [consecutive_cosines[i] for i in sentence_cut_ids],
        color="red",
        marker="x",
    )
    plt.vlines(
        [s + 0.5 for s in sentence_cut_ids],
        ymin=0,
        ymax=1,
        color="red",
        linestyles="dashed",
    )
    plt.title("Cosine similarity between consecutive segments")

    # plot the cosine similarity matrix
    plt.subplot(2, 1, 2)
    c = cosine_similarity(sentence_embeddings, sentence_embeddings)
    plt.imshow(c, cmap="Blues")
    plt.yticks(
        ticks=np.arange(num_sentences),
        labels=[s.replace("\n", "") for s in sentence_splits],
    )
    plt.colorbar()
    plt.scatter(sentence_cut_ids, sentence_cut_ids, color="red", marker="x")
    plt.hlines(
        y=[s + 0.5 for s in sentence_cut_ids],
        xmin=-0.5,
        xmax=num_sentences - 1,
        color="red",
        linestyles="dashed",
    )
    plt.title("Cosine similarity matrix")


funcs = [
    sentence_splits_ids,
    filtered_sentence_split_ids,
    sentence_splits,
    sentence_embeddings,
    consecutive_cosines,
    sentence_num_tokens,
    sentence_cut_ids,
    segment_keys,
    text_segments,
    # meaningful_sentences,
    # sentence_cut_ids,
    # sentence_chunk_ids,
    # character_chunk_ids,
    # chunk_text,
    display_cut_ids,
]
segmentation_dag = DAG(funcs)


def character_chunker(text, max_chunk_size):
    splits_ids = sentence_splits_ids(text)
    sentences = sentence_splits(text, splits_ids)
    num_tokens = sentence_num_tokens(sentences)
    segment_keys = []
    start = splits_ids[0][0]
    stop = splits_ids[0][1]
    numtok = num_tokens[0]
    n_sentences = len(splits_ids)
    for i in range(1, n_sentences):
        if numtok + num_tokens[i] > max_chunk_size:
            segment_keys.append((start, stop))
            numtok = num_tokens[i]
            start = splits_ids[i][0]
            stop = splits_ids[i][1]
        else:
            numtok += num_tokens[i]
            stop = splits_ids[i][1]
    segment_keys.append((start, stop))
    return segment_keys
