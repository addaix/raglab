from stores.repos import RepoJson
from py2http import mk_flat, run_app


class FiddleManager:
    def __init__(self):
        self.repo = RepoJson()

    def create(self, user_id: str, name: str):
        print("Creating object")
        self.repo.create(user_id + name, "")

    def get(self, user_id: str, name: str) -> str:
        return self.repo.get(user_id + name)

    def update(self, user_id: str, name: str, value: str):
        return self.repo.update(user_id + name, value)

    def delete(self, user_id: str, name: str):
        return self.repo.delete(user_id + name)


def run_fiddle_manager():
    create = mk_flat(FiddleManager, FiddleManager.create, func_name="create")
    get = mk_flat(FiddleManager, FiddleManager.get, func_name="get")
    update = mk_flat(FiddleManager, FiddleManager.update, func_name="update")
    delete = mk_flat(FiddleManager, FiddleManager.delete, func_name="delete")
    fns = [create, get, update, delete]

    run_app(fns, publish_openapi=True, publish_swagger=True)
