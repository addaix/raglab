FROM python:3.9

WORKDIR /app

COPY . /app

COPY .env /app
COPY test.db /app

RUN pip install fastapi gpt4all langchain chromadb lkj config2py

RUN pip install lkj
RUN pip install config2py

CMD ["python", "app.py"]
