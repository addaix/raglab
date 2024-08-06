from py2http.service import mk_app

from sqlalchemy import engine_from_config, create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqldol.base import SqlBaseKvReader

import config2py

DATABASE_URL = config2py.config_getter("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_user_id_by_token(token):
    user = SqlBaseKvReader(
        engine=engine_from_config,
        table_name="users",
        key_columns="token",
        value_columns=["id"],
    )
    user_id = user[token]
    return user_id


class PromptPilote:

    def list(token: str, db: Session = get_db()):
        user_id = get_user_id_by_token(token)

        permissions = SqlBaseKvReader(
            engine=engine,
            table_name="app_permission",
            key_columns="id",
            value_columns=["app_id", "user_id"],
        )

        permissions_id = []

        for id in permissions:
            if permissions[id]["user_id"] == user_id:
                permissions_id.append(permissions[id]["app_id"])

        apps = SqlBaseKvReader(
            engine=engine, table_name="app", key_columns="id", value_columns=["name"]
        )
        app_names = [apps[id]["name"] for id in apps]
        return {"names": app_names}


app = mk_app(
    [
        dict(
            endpoint=PromptPilote,
            name="promptpilote",
            attr_names=["list"],
        ),
    ],
    publish_openapi=True,
    publish_swagger=True,
)

if __name__ == "__main__":
    app.run()
