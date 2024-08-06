import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv


class ChromaClient:

    def __init__(self):
        load_dotenv()
        chroma_key = os.getenv("CHROMA_KEY")
        self.client = chromadb.HttpClient(
            host="149.202.47.109",
            port="45000",
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials=chroma_key,
            ),
        )

    def get_collections(self):
        collections = self.client.list_collections()
        return collections

    def add_collection(self, name: str):
        self.client.create_collection(name)

    def query(self, dataset_name: str, query: str):
        collection = self.client.get_collection(dataset_name)
        return collection.query(query_texts=query)
