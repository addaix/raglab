from sqlalchemy import create_engine, Table, MetaData, delete, select
from typing import Mapping, Sized, Iterable
from chromadb import Client, Settings, HttpClient

from langchain_community.llms.openai import OpenAIChat
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.prompts import ChatPromptTemplate

import json

import config2py

POSTGRESS_TEST_DB_URL = config2py.config_getter('POSTGRESS_TEST_DB_URL')

class PostgresTableRows(Sized, Iterable):
    def __init__(self, engine, table_name):
        self.engine = engine
        self.table_name = table_name
        self.metadata = MetaData()
        self.table = Table(self.table_name, self.metadata, autoload_with=self.engine)

    def __iter__(self):
        query = select(self.table)
        with self.engine.connect() as connection:
            result = connection.execute(query)
            for row in result:
                yield row

    def __len__(self):
        query = select(self.table)
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result.rowcount
        
class PostgresBaseColumnsReader(Mapping):
    """Here, keys are column names and values are column values"""
    def __init__(self, engine, table_name):
        self.engine = engine
        self.table_name = table_name
        self.metadata = MetaData()
        self.table = Table(self.table_name, self.metadata, autoload_with=self.engine)
        
    def __iter__(self):
        return (column_obj.name for column_obj in self.table.columns)
    
    def __len__(self):
        return len(self.table.columns)
    
    def __getitem__(self, key):
        # TODO: Finish
        query = select(self.table).with_only_columns([self.table.c[key]])
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result.fetchall()
    

from typing import Callable  


class PostgresBaseKvReader(Mapping):
    """A mapping view of a table, 
    where keys are values from a key column and values are values from a value column.
    There's also a filter function that can be used to filter the rows.
    """
    def __init__(
            self, engine, table_name, 
            key_columns=None, 
            value_columns=None,
            filt=None
        ):
        self.engine = engine
        self.table_name = table_name
        self.metadata = MetaData()
        self.table = Table(self.table_name, self.metadata, autoload_with=self.engine)
        self.filt = filt

    def __iter__(self):
        query = select(self.table)
        with self.engine.connect() as connection:
            result = connection.execute(query)
            for row in result:
                yield row

    def __len__(self):
        query = select(self.table)
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result.rowcount
        
    def __getitem__(self, key):
        query = select(self.table)
        with self.engine.connect() as connection:
            result = connection.execute(query)
            return result.fetchall()
        
        
class PostgressTables(Mapping):
    def __init__(self, engine):
        self.engine = engine
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def __getitem__(self, key):
        return PostgresBaseKvReader(self.engine, key)
        # Or do something with this:
        # return self.metadata.tables[key]

    def __iter__(self):
        return iter(self.metadata.tables)

    def __len__(self):
        return len(self.metadata.tables)


import os
import re
from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from raglab2 import Raglab2, RaglabSessionBuilder, RaglabSession

from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Déclaration de la base pour la définition des modèles SQLAlchemy
Base = declarative_base()

class User(Base) :
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    token = Column(String, index=True)

class App(Base) :
    __tablename__ = "app"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class AppPermission(Base) :
    __tablename__ = "app_permission"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer)
    app_id = Column(Integer)

class PromptTemplate(Base) :
    __tablename__ = "prompt_template"

    id = Column(Integer, primary_key=True)

    name = Column(String)
    template = Column(String)
    rjsf_ui = Column(String)

running_sessions = dict[str, RaglabSession]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
app.secret_key =  config2py.config_getter("RAGLAB_API_KEY")
origins = ["*"]
app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

auth_layer = APIKeyHeader(name="API_TOKEN")



@app.get("/app_list")
def app_list(token:str=Depends(auth_layer), db:Session = Depends(get_db)) :  
    result = db.query(User, AppPermission, App).filter(User.token == token).join(AppPermission, User.id == AppPermission.user_id).join(App, App.id == AppPermission.app_id).all()
    app_names = [r[2].name for r in result]
    return {
        "names" : app_names
    }

@app.get("/prompt")
def get_template_list(token:str=Depends(auth_layer), db:Session = Depends(get_db)) :
    result = db.query(PromptTemplate).all()

    lst = []

    for r in result :
        lst.append({
            "name" : r.name,
            "template" : r.template
        })

    return lst

@app.get("/prompt/editor")
def get_editor(name:str, token:str=Depends(auth_layer), db:Session=Depends(get_db)) :
    result = db.query(PromptTemplate).filter(PromptTemplate.name == name).first()

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
        dico[p] = {'type': 'string', 'title': f'{p}'}

    prompt_template.rjsf_ui = json.dumps({
            'title': 'Prompt arguments', 
            'type': 'object', 
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

def _default_chat():
    from oa import chat

    return chat

from oa import prompt_function
from i2 import Sig

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


@app.get("/app/{name}/")
def run_app(name:str, token:str=Depends(auth_layer), db:Session = Depends(get_db)) :
    result = db.query(App).filter(App.name == name).first()

    #### TODO: Build from App ####
    chroma_key = config2py.config_getter("CHROMA_KEY")
    client = HttpClient(host='149.202.47.109', port="45000", settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider", chroma_client_auth_credentials=chroma_key))
    embedding = GPT4AllEmbeddings(client=client)
    db = Chroma(collection_name="poc", client=client, embedding_function=embedding)
    llm = GPT4All("all-MiniLM-L6-v2-f16.gguf")
    prompt_template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    ######## END TODO ########

    session = RaglabSessionBuilder.build(
        vectorestore=db,
        embedding=embedding,
        prompt=prompt,
        model=llm
    )

    running_sessions[name] = session

    return 200

@app.post("/app/{name}")
def ask_app(name:str, query:str) :
    if name in running_sessions :
        return running_sessions[name].ask(query)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)