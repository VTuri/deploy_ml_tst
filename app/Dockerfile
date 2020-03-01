FROM python:3.7.6-slim-stretch

COPY deploy_ml /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD flask run --host=0.0.0.0
