import config2py
from http.client import HTTPException
from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from i2 import Sig
import json
from pydantic import BaseModel
import oa
from oa import chat, prompt_function
import re
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker, Session

from sqldol.stores import SqlDictReader, SqlDictStore
import uvicorn
from dotenv import load_dotenv

load_dotenv()


DATABASE_URL = config2py.config_getter("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()
app.secret_key = config2py.config_getter("RAGLAB_API_KEY")
origins = ["http://localhost:3000", "https://raglab-ui.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache_token_user_id = {}

auth_layer = APIKeyHeader(name="API_TOKEN")


def check_auth_layer(token: str = Depends(auth_layer)):
    if token in cache_token_user_id:
        return True

    raise HTTPException(
        status_code=401,
        detail="Invalid API Key",
    )


def get_auth_id_layer(token: str = Depends(auth_layer)):
    user = SqlDictReader(
        engine=engine, table_name="users", key_columns="token", value_columns=["id"]
    )
    if token in user:
        user_id = user[token]["id"]
        cache_token_user_id[token] = user_id
        return int(user_id)

    raise HTTPException(
        status_code=401,
        detail="Token not found",
    )


@app.get("/app_list")
def app_list(user_id=Depends(get_auth_id_layer)):
    print(f"User {user_id} wants to select an app")
    permissions = SqlDictReader(
        engine=engine,
        table_name="app_permission",
        key_columns="id",
        value_columns=["app_id", "user_id"],
    )

    permissions_id = []

    for id in permissions:
        if permissions[id]["user_id"] == user_id:
            permissions_id.append(permissions[id]["app_id"])

    print(f"User can access {len(permissions_id)} app")
    apps = SqlDictReader(
        engine=engine, table_name="app", key_columns="id", value_columns=["name"]
    )

    app_names = [apps[id]["name"] for id in permissions_id]

    return {"names": app_names}


@app.get("/prompt")
def get_template_list(user_id=Depends(get_auth_id_layer)):
    print(f"User {user_id} wants to get prompt templates")

    store = SqlDictReader(
        engine=engine,
        table_name="prompt_template",
        key_columns="id",
        value_columns=["name", "template", "owner_id"],
    )

    templates = []
    for key in store:
        if store[key]["owner_id"] == user_id:
            templates.append(store[key])

    lst = [{"name": t["name"], "template": t["template"]} for t in templates]
    return lst


@app.get("/prompt/editor")
def get_editor(name: str):
    store = SqlDictReader(
        engine=engine,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["name", "template", "rjsf_ui"],
    )

    try:
        result = store[name]
    except:
        return Response(status_code=404)

    return {
        "prompt": {"name": result["name"], "template": result["template"]},
        "rjsf_ui": json.loads(result["rjsf_ui"]),
    }


class SavePromptTemplateRequest(BaseModel):
    name: str
    template: str


@app.post("/prompt/editor")
def save_prompt(request: SavePromptTemplateRequest, user_id=Depends(get_auth_id_layer)):
    print(f"User {user_id} wants to save {request.name}")

    store = SqlDictStore(
        engine=engine,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["template", "rjsf_ui", "owner_id"],
    )

    dico = {}
    for p in parse_parameters(request.template):
        dico[p] = {"type": "string", "title": p}

    prompt_template = {
        "name": request.name,
        "template": request.template,
        "rjsf_ui": json.dumps(
            {
                "title": "Prompt arguments",
                "type": "object",
                "required": list(dico.keys()),
                "properties": dico,
            }
        ),
        "owner_id": user_id,
    }

    store[request.name] = prompt_template

    return {
        "prompt": {
            "name": prompt_template["name"],
            "template": prompt_template["template"],
        },
        "rjsf_ui": {
            "title": "Prompt arguments",
            "type": "object",
            "required": list(dico.keys()),
            "properties": dico,
        },
    }


@app.delete("/prompt/editor")
def delete_prompt_template(
    keys: list[str], user_id=Depends(get_auth_id_layer), db: Session = Depends(get_db)
):
    print(f"User {user_id} wants to save {keys}")

    store = SqlDictStore(
        engine=engine,
        table_name="prompt_template",
        key_columns="name",
        value_columns=["owner_id"],
    )

    for key in keys:
        if key in store:
            if store[key]["owner_id"] == user_id:
                store.__delitem__(key)

    return Response(status_code=200)


class AskPromptRequest(BaseModel):
    template: str
    parameters: list[str]


def parse_parameters(template: str) -> list[str]:
    pattern = r"\{([^}]*)\}"
    return re.findall(pattern, template)


def _default_chat():
    return chat


def prompt_execution_adapter(prompt_template, params):
    f = prompt_function(prompt_template)
    sig = Sig(f)
    kwargs = sig.map_arguments(params, ignore_kind=True)
    return f(**kwargs)


@app.post("/prompt/editor/ask")
def ask_prompt(request: AskPromptRequest):
    return {"answer": prompt_execution_adapter(request.template, request.parameters)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="trace")
