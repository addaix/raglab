import os
import re
from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from raglab2 import RaglabSessionBuilder, RaglabSession

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

load_dotenv()
app = FastAPI()
app.secret_key =  os.getenv("RAGLAB_API_KEY")
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
            "id" : r.id,
            "name" : r.name,
            "content" : r.template
        })

    return lst

@app.get("/prompt/editor")
def get_editor(id:str, db:Session=Depends(get_db)) :
    result = db.query(PromptTemplate).filter(PromptTemplate.id == id).first()

    if result is None :
        return Response(status_code=404)

    return {
        "prompt" : {
            "id": result.id,
            "name" : result.name,
            "template": result.template
        },
        "rjsf_ui" : result.rjsf_ui
    }

class SavePromptTemplateRequest(BaseModel) :
    id: int
    name: str
    template: str

def get_rjsf_from_template(template:str) :
    parameters = parse_parameters(template)
    schema = """
{
    'title': 'Prompt arguments', 
    'type': 'object', 
    'properties':
""" + ",".join("{'type': 'string', 'title': '{p}'}" for p in parameters) + "} }"
    return schema

@app.post("/prompt/editor")
def save_prompt(request:SavePromptTemplateRequest, db:Session=Depends(get_db)) :    
    prompt_template = db.query(PromptTemplate).filter(PromptTemplate.id == request.id).first()

    if prompt_template is None :
        prompt_template = PromptTemplate()

    prompt_template.name = request.name
    prompt_template.template = request.template
    prompt_template.rjsf_ui = get_rjsf_from_template(request.template)
    db.add(prompt_template)
    db.commit()
    return Response(status_code=200)


from chromadb import Client, Settings

from langchain_community.llms.openai import OpenAIChat
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

class AskPromptRequest(BaseModel) :
    template:str
    parameters:list[str]

def parse_parameters(template:str) -> list[str] :
    pattern = r'\{([^}]*)\}'
    return re.findall(pattern, template)

@app.post("/prompt/editor/ask")
def ask_prompt(request:AskPromptRequest, token:str=Depends(auth_layer)) :
    model = OpenAIChat(model_name="gpt-3.5-turbo")

    parameters = parse_parameters(request.template)
    i = 0
    for p in parameters :        
        request.template = request.template.replace("{"+p+"}", request.parameters[i])
        i += 1

    return model.ask(request.template)

@app.get("/app/{name}/")
def run_app(name:str, token:str=Depends(auth_layer), db:Session = Depends(get_db)) :
    result = db.query(App).filter(App.name == name).first()

    #### TODO: Build from App ####

    client = Client(host='149.202.47.109', port="45000", settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider", chroma_client_auth_credentials=os.getenv("CHROMA_KEY")))
    embedding = GPT4AllEmbeddings(client=client)
    db = Chroma(collection_name="name", client=client, embedding_function=embedding)
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