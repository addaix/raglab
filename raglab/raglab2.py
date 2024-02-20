"""
Taking the use case implemented by example: 
    https://python.langchain.com/docs/expression_language/cookbook/retrieval
I created a class that encapsulates the logic of the example, and added some
features to it. The class is called Raglab2.


Example of use:

```python
from srag.tw.raglab2 import *
raglab = raglab2('test')

text_store = {
    "doc1": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
    "doc2": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China.",
    "doc3": "The light bulb was invented by Thomas Edison in the 19th century. It was a groundbreaking invention that allowed for the widespread use of electric light in homes and businesses.",
    "doc4": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose, which can be used to fuel the organism's activities."
}

# text_store = wrap_in_langchain_document(text_store)

raglab.update_with_mapping(text_store)

raglab.ask("Who invented the light bulb and what is the process plants use to convert light energy?")
```

"""

from typing import Optional, Callable, Mapping, Union
from functools import cached_property, partial
import os
import chromadb
from chromadb.config import Settings

from dotenv import load_dotenv

from lkj import import_object, chunker, clog  # pip install lkj
from config2py import get_configs_folder_for_app  # pip install config2py

from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.llms.gpt4all import GPT4All
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

segmenter = partial(chunker, chk_size=1000, include_tail=True)

DFLT_VECTORSTORES_PERSIST_ROOTDIR = get_configs_folder_for_app(
    'srag', configs_name='vectorstores'
)

DFLT_PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

DFLT_PROMPT = ChatPromptTemplate.from_template(DFLT_PROMPT_TEMPLATE)


class Raglab2:
    def __init__(
        self,
        vectorstore: Union[str, VectorStore] = 'Chroma',
        embedding='GPT4AllEmbeddings',
        prompt=DFLT_PROMPT,
        model=GPT4All(model="orca-mini-3b-gguf2-q4_0"), #all-MiniLM-L6-v2-f16.gguf orca-mini-3b-gguf2-q4_0.gguf mistral-7b-openorca.Q4_0.gguf
    ):
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.prompt = prompt
        self.model = model

    def update_with_mapping(
        self,
        mapping: Mapping[str, str],
        *,
        chk_size: int = 1000,
        verbose: bool = False,
        # filt: Callable = None  # to control update rules
    ):

        ids, texts = map(list, zip(*mapping.items()))
        self.vectorstore.from_texts(texts=texts, ids=ids, embedding=self.embedding)

        # TODO: Make the segmented (batched) update of the vectorstore work
        #  --> Need to find the right vectorstore method for that (maybe update_documents?)
        # TODO: Figure out how to only update vectorstores when "docs" have changed
        #   (editted, deleted, added, etc.)
        # _segmenter = partial(segmenter, chk_size=chk_size)
        # _clog = clog(verbose, log_func=lambda x: print(f"segment {i}"))

        # for i, _items in enumerate(_segmenter(mapping.items())):
        #     _clog(i)
        #     ids, texts = map(list, zip(*_items))
        #     self.vectorstore.from_texts(texts=texts, ids=ids, embedding=self.embedding)

        return self

    @cached_property
    def retriever(self):
        return self.vectorstore.as_retriever()

    @cached_property
    def chain(self):
        return (
            {"context": self.retriever , "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        return self.chain.invoke(query)     


from langchain.docstore.document import Document
from dol import wrap_kvs


def _wrap_kv_in_langchain_document(k, v):
    return Document(page_content=v, metadata={'key': k})


def wrap_in_langchain_document(mapping: Mapping[str, str]):
    return wrap_kvs(mapping, postget=_wrap_kv_in_langchain_document)


def object_resolver(module_dot_path, obj_name=None, *args, **kwargs):
    if obj_name is None:
        return partial(object_resolver, module_dot_path)
    return import_object('.'.join([module_dot_path, obj_name]))

def raglab2(
    dataset_name: str,
    *,
    vectorstore: Union[str, VectorStore] = 'Chroma',
    embedding: str = 'GPT4AllEmbeddings',
    persist_rootdir: Optional[str] = DFLT_VECTORSTORES_PERSIST_ROOTDIR,
    vector_stores_module_dot_path='langchain_community.vectorstores.chroma',
    embeddings_module_dot_path='langchain.embeddings',
) -> Raglab2 :
    """
    Create a Raglab2 instance with the given parameters.
    This is a convenience function to create a Raglab2 instance with the string
    parameters.

    This means it can directly be exposed to a CLI, http server, and GUI.
    """
    _construction_kwargs = locals()
    mk_embeddings = object_resolver(embeddings_module_dot_path)
    # TODO: Add if dataset is None, list dataset names

    load_dotenv()
    chroma_key = os.getenv('CHROMA_KEY')
    client = chromadb.HttpClient(host='149.202.47.109', port="45000", settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider", chroma_client_auth_credentials=chroma_key))
    embedding = GPT4AllEmbeddings(client=client)
    db = Chroma(collection_name=dataset_name, client=client, embedding_function=embedding)
    raglab = Raglab2(db, embedding)

    raglab._construction_kwargs = _construction_kwargs
    return raglab

