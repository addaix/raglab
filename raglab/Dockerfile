FROM python:3.9

WORKDIR /app

COPY . /app

COPY .env /app

RUN pip install flask gpt4all langchain chromadb lkj config2py

RUN pip install lkj
RUN pip install config2py

CMD ["flask", "run", "--host=0.0.0.0"]
