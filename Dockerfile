FROM python:3.7-slim-buster
WORKDIR /app
COPY . /app

RUN apt-get update -y && apt-get install -y awscli

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
