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
from meshed import DAG
import matplotlib.pyplot as plt

OPENAI_API_KEY = config_getter("OPENAI_API_KEY")

# # Download the necessary NLTK data files (only needed once)
# nltk.download("punkt")
# # Segment the text into sentences
# sentence_splits = sent_tokenize
# sentence_splits.__name__ = "sentence_splits"


def sentence_splits_ids(text, sep=[".", "!", "?"]):
    indices = []
    start = 0
    for i, char in enumerate(text):
        if char in sep:
            indices.append((start, i + 1))
            start = i + 1
    if start < len(text):
        indices.append((start, len(text)))
    return indices


def sentence_splits(text, sentence_splits_ids):
    return [text[start:stop] for start, stop in sentence_splits_ids]


# def is_meaningful(sentence, language="english"):
#     stop_words = set(stopwords.words(language))
#     words = word_tokenize(sentence)
#     # Filter out punctuation and stopwords
#     meaningful_words = [
#         word
#         for word in words
#         if word.lower() not in stop_words and word not in punctuation
#     ]
#     return len(meaningful_words) > 0


# def meaningful_sentences(sentence_splits):
#     return [sentence for sentence in sentence_splits if is_meaningful(sentence)]


def sentence_embeddings(sentence_splits, api_key=OPENAI_API_KEY, dimension=512):
    embeddings_model = OpenAIEmbeddings(
        api_key=api_key, model="text-embedding-3-small", dimensions=dimension
    )
    return np.array(embeddings_model.embed_documents(sentence_splits))


def consecutive_cosines(sentence_embeddings):
    z = zip(sentence_embeddings, sentence_embeddings[1:])
    cosines = [cosine_similarity([a], [b])[0][0] for a, b in z]
    return cosines


def sentence_cut_ids(consecutive_cosines, **kwargs):
    """
    Returns the indices of the sentences to cut. The cut sentence belong to the next segment.
    kwargs: height=None, threshold=None, distance=None,
            prominence=None, width=None, wlen=None, rel_height=0.5,
            plateau_size=None"""
    neg_cosines = [-c for c in consecutive_cosines]
    peaks_idx, _ = find_peaks(neg_cosines, **kwargs)
    return [peak + 1 for peak in peaks_idx]  # +1 to account for the shift (start at 0)


# def sentence_chunk_ids(cut_ids):
#     segments_ids = []
#     start = 0
#     for stop in cut_ids:
#         segments_ids.append((start, stop + 1))
#         start = stop + 1
#     segments_ids.append((start, -1))
#     return segments_ids


def character_chunk_ids(sentence_cut_ids, sentence_splits_ids):
    character_chunk_ids = []
    start_character = 0
    for stop_sentence in sentence_cut_ids:
        stop_character = sentence_splits_ids[stop_sentence][1]
        character_chunk_ids.append((start_character, stop_character))
        start_character = stop_character
    return character_chunk_ids


# def character_chunk_ids(sentence_cut_ids, sentence_splits):
#     """
#     >>> sentence_splits = ['This is a sentence.', 'This is another sentence.', "Now here is a third sentence."]
#     >>> sentence_cut_id = [1]
#     >>> character_chunk_ids = character_chunk_ids(sentence_chunk_ids, sentence_splits)
#     >>> character_chunk_ids
#     [(0, 20), (20, 60)]
#     """
#     print(sum([len(sentence) for sentence in sentence_splits]))
#     character_chunk_ids = []
#     start_sentence = 0
#     start_character = 0
#     for stop_sentence in sentence_cut_ids:
#         stop_character = start_character + len(
#             " ".join(sentence_splits[start_sentence:stop_sentence])
#         )
#         character_chunk_ids.append((start_character, stop_character))
#         start_character = stop_character
#         start_sentence = stop_sentence
#     return character_chunk_ids


def chunk_text(character_chunk_ids, text):
    print(len(text))
    return [text[start:stop] for start, stop in character_chunk_ids]


def display_cut_ids(
    sentence_embeddings, consecutive_cosines, sentence_cut_ids, verbose=True
):
    if not verbose:
        return
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
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
    plt.subplot(1, 2, 2)
    c = cosine_similarity(sentence_embeddings, sentence_embeddings)
    plt.imshow(c, cmap="Blues")
    plt.colorbar()
    plt.scatter(sentence_cut_ids, sentence_cut_ids, color="red", marker="x")
    plt.title("Cosine similarity matrix")


funcs = [
    sentence_splits_ids,
    consecutive_cosines,
    sentence_embeddings,
    # meaningful_sentences,
    sentence_cut_ids,
    # sentence_chunk_ids,
    character_chunk_ids,
    sentence_splits,
    chunk_text,
    display_cut_ids,
]
segmentation_dag = DAG(funcs)


def segment_keys(documents, api_key=OPENAI_API_KEY, language="english", **kwargs):
    segment_keys = []
    for doc_name, text in documents.items():
        character_chunk_ids = segmentation_dag[:"character_chunk_ids"](
            language=language, text=text, api_key=api_key, **kwargs
        )
        for start, stop in character_chunk_ids:
            segment_keys.append((doc_name, start, stop))
    return segment_keys
