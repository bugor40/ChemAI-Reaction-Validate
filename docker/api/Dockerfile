FROM python:3.11.9-alpine3.19

COPY ./ /app/

RUN cd app && pip install -r requirements.txt

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]