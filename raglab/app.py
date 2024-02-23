from flask import Flask, Response, request, redirect, session, url_for, render_template
from raglab.retrieval.chroma_client import ChromaClient
from raglab2 import *

app = Flask(__name__)
load_dotenv()

app.secret_key =  os.getenv("RAGLAB_API_KEY")

USER = "user"
PASSWORD = "mdp"

### AUTH

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == USER and password == PASSWORD:
        # TODO SECURITY THINGS 
        return redirect(url_for(chat))

    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/gate')
def gate():
    return render_template('gate.html')

### RAGLAB

@app.route('/raglab')
def index():
    actions = {
            "/raglab/query" : "Query dataset",
            "/raglab/search" : "To be continued...",
        }

    return render_template('raglab_home.html', actions=actions)

lab = raglab2("default")

@app.route("/raglab/chat", methods=["GET", "POST"])
def chat() :
    if(request.method == "GET") :
        return render_template("chat.html")
    
    elif request.method == "POST" : 
        message = request.json["message"]
        response = lab.ask(message)
        return Response(status=200, response=response)
    
    else :
        return Response(status=666)
    
@app.route("/raglab/chat/dataset", methods=["POST"])
def chat_change_dataset(name) :
    lab = raglab2(name)
    message = request.json["message"]
    response = lab.ask(message)
    return Response(status=200, response=response)

@app.route("/raglab/datasets", methods=["GET"])
def get_datasets() :
    client = ChromaClient()

    datasets = []
    for d in client.get_collections() :
        datasets.append(d.name)

    response = { "datasets" : datasets }
    return Response(status=200, response=response)

@app.route("/raglab/datasets", methods=["POST"])
def get_datasets() :
    client = ChromaClient()

    datasets = []
    for d in client.get_collections() :
        datasets.append(d.name)

    response = { "datasets" : datasets }
    return Response(status=200, response=response)

@app.route('/raglab/query')
def get_query():
    service = None
    result = service.datasets()

    return render_template('raglab_query.html', elements=result.results)

@app.route('/raglab/query', methods=['POST'])
def post_query():
    query = request.form['query']
    name = request.form['element']

    service = None
    query_result = service.query(name, query)

    return render_template("raglab_query_result.html", result=query_result.results)

@app.route('/raglab/search')
def search():
    return f'<h1>In progress...</h1>'

if __name__ == '__main__':
    app.run(debug=True)
