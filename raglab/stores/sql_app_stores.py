"""Definition of SQL app stores."""

from functools import partial
from i2 import Sig, Namespace
from sqldol.base import SqlBaseKvReader
from sqldol.stores import SqlDictReader

from raglab.util import LazyAccessor

store_factories = Namespace(
    permissions=partial(
        SqlDictReader,
        table_name="app_permission",
        key_columns="id",
        value_columns=["app_id", "user_id"],
    ),
    permissions_by_app_id=partial(
        SqlDictReader,
        table_name="app_permission",
        key_columns="app_id",
        value_columns=["app_id", "user_id"],
    ),
    apps=partial(
        SqlDictReader, table_name="app", key_columns="id", value_columns=["name"]
    ),
    ided_templates=partial(
        SqlDictReader,
        table_name="prompt_template",
        key_columns="id",
        value_columns=["name", "template", "owner_id"],
    ),
    templates_and_uis=partial(
        SqlDictReader,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["name", "template", "rjsf_ui"],
    ),
    templates_and_uis_with_owner_id=partial(
        SqlDictReader,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["template", "rjsf_ui", "owner_id"],
    ),
    owner_ids=partial(
        SqlDictReader,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["owner_id"],
    ),
)

mk_mall = partial(LazyAccessor, store_factories)

# give it a more helpful signature:
mk_mall.__signature__ = Sig("(*, engine)")
