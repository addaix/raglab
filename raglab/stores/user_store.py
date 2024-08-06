from raglab.util import StoreAccess
from raglab.stores import stores_util


class UserStore(StoreAccess):
    def __init__(self):
        self.store = stores_util.mk_json_store()
