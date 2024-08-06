from stores import stores_util

from dol import *


class Repo:
    def __init__(self):
        wrap_kvs
        store = self.get_store()
        self.store = store  # TODO : Implement object wrapper
        self.key_template = self.get_key_template()

    def get_store(self) -> Store:
        raise Exception("This method should be implemented.")

    def get_key_template(self) -> KeyTemplate:
        raise Exception("This method should be implemented.")

    def create(self, key: str, value: str):
        self.store[key] = value

    def get(self, key: str) -> str:
        return self.store[key]

    def update(self, key: str, value: str) -> Store:
        self.store[key] = value

    def delete(self, key: str):
        self.store.remove[key]


class RepoJson(Repo):
    def get_store(self):
        return stores_util.mk_json_store(".")

    def get_key_template(self) -> stores_util.KeyTemplate:
        return KeyTemplate(self.store)
