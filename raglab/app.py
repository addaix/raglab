
import os
import re
from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from raglab2 import Raglab2, RaglabSessionBuilder, RaglabSession

from sqlalchemy import create_engine, Table, MetaData, delete, select
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from sqldol.stores import SqlDictsReader

from typing import Mapping, Sized, Iterable

import json

import config2py

DATABASE_URL = config2py.config_getter('DATABASE_URL')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

running_sessions = dict[str, RaglabSession]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.secret_key =  config2py.config_getter("RAGLAB_API_KEY")
origins = ["http://localhost:3000", "https://raglab-ui.vercel.app"]
app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

auth_layer = APIKeyHeader(name="API_TOKEN")

def get_user_id_by_token(token) :
    user = SqlDictsReader(engine=engine, table_name="users", key_columns="token", value_columns=["id"])
    user_id = user[token]
    return user_id

@app.get("/app_list")
def app_list(token:str=Depends(auth_layer), db:Session = Depends(get_db)) :  
    user_id = get_user_id_by_token(token)
    
    permissions = SqlDictsReader(
        engine=engine, 
        table_name="app_permission", 
        key_columns="id", 
        value_columns=["app_id", "user_id"])
    
    permissions_id = []

    for id in permissions :
        if permissions[id]["user_id"] == user_id:
            permissions_id.append(permissions[id]["app_id"])

    apps = SqlDictsReader(engine=engine, table_name="app", key_columns="id", value_columns=["name"])
    app_names = [app[id]["name"] for id in apps]
    return {
        "names" : app_names
    }

@app.get("/prompt")
def get_template_list(token:str=Depends(auth_layer), db:Session = Depends(get_db)) :
    user_id = get_user_id_by_token(token)
    
    store = SqlDictsReader(
        engine=engine, 
        table_name="prompt_template", 
        key_columns="id", 
        value_columns=["name", "template", "owner_id"])

    templates = []
    for key in store :
        if store[key]["owner_id"] == user_id :
            templates.add(store[key])

    lst = [{"name": t["name"], "template": t["name"]} for t in templates]
    return lst

@app.get("/prompt/editor")
def get_editor(name:str, token:str=Depends(auth_layer), db:Session=Depends(get_db)) :
    store = SqlDictsReader(
        engine=engine, 
        table_name="prompt_template", 
        key_columns="name", 
        value_columns=["name"])

    if result is None :
        return Response(status_code=404)

    print

    return {
        "prompt" : {
            "name" : result.name,
            "template": result.template
        },
        "rjsf_ui" : json.loads(result.rjsf_ui)
    }

class SavePromptTemplateRequest(BaseModel) :
    name: str
    template: str

def get_rjsf_from_template(template:str) :
    parameters = parse_parameters(template)
    json_schema = """
{
    'title': 'Prompt arguments', 
    'type': 'object', 
    'properties':{
""" + ",".join(f"\'{p}\' : {{'type': 'string', 'title': '{p}'}}" for p in parameters) + "}}"
    
    #ui_schema ="{"+",".join([f"\"{p}\" : {{\"ui:widget\": \"textarea\"}}" for p in parameters])+"}"

    return json_schema



@app.post("/prompt/editor")
def save_prompt(request:SavePromptTemplateRequest, token:str=Depends(auth_layer), db:Session=Depends(get_db)) :    
    print(f"User {token} wants to save {request.name}")

    prompt_template = db.query(PromptTemplate).filter(PromptTemplate.name == request.name).first()

    if prompt_template is None :
        prompt_template = PromptTemplate()

    prompt_template.name = request.name
    prompt_template.template = request.template
    

    dico = {}

    for p in parse_parameters(request.template) : 
        dico[p] = {'type': 'string', 'title': p}

    prompt_template.rjsf_ui = json.dumps({
            'title': 'Prompt arguments', 
            'type': 'object', 
            'required' : list(dico.keys()),
            'properties': dico
        })

    db.add(prompt_template)
    db.commit()
    
    return {
        "prompt" : {
            "name" : prompt_template.name,
            "template": prompt_template.template
        },
        "rjsf_ui" : {
            'title': 'Prompt arguments', 
            'type': 'object', 
            'required': list(dico.keys()),
            'properties': dico
        }
    }

@app.delete("/prompt/editor")
def delete_prompt_template(keys:list[str], token:str=Depends(auth_layer), db:Session=Depends(get_db)) :

    print(f"User {token} wants to delete {keys}")

    for key in keys :
        db.execute(delete(PromptTemplate).where(PromptTemplate.name == key))
        print(f"Deleted {key}")
    db.commit()
    return Response(status_code=200)

class AskPromptRequest(BaseModel) :
    template:str
    parameters:list[str]

def parse_parameters(template:str) -> list[str] :
    pattern = r'\{([^}]*)\}'
    return re.findall(pattern, template)

import oa
from oa import prompt_function
from i2 import Sig

def _default_chat():
    from oa import chat

    return chat

def prompt_execution_adapter(prompt_template, params):
    f = prompt_function(prompt_template)
    sig = Sig(f)
    kwargs = sig.map_arguments(params, ignore_kind=True)
    return f(**kwargs)

@app.post("/prompt/editor/ask")
def ask_prompt(request:AskPromptRequest, token:str=Depends(auth_layer)) :
    return {
        "answer": prompt_execution_adapter(request.template, request.parameters)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="trace")