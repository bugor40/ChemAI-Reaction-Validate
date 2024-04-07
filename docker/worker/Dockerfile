FROM python:3.11.9-alpine3.19

COPY ./ /app/

RUN cd app && pip install -r requirements.txt

WORKDIR /app

ENTRYPOINT celery -A main.celery worker --loglevel=INFO