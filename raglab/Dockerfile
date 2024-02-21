FROM python:3.9

WORKDIR /app

COPY . /app

COPY .env /app

RUN pip install flask gpt4all langchain chroma lkj config2py

RUN pip install lkj
RUN pip install config2py

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["python", "app.py"]
