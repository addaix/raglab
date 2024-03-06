FROM python:3.9

WORKDIR /app

COPY . /app

COPY .env /app
COPY test.db /app

RUN pip install fastapi gpt4all openai langchain chromadb lkj config2py dill dol tiktoken oa ju

CMD ["python", "app.py"]
