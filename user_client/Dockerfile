FROM python:3.7-alpine

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "user_client.request_base:app", "--host", "0.0.0.0", "--port", "80"]