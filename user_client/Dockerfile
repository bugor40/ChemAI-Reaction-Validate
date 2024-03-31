FROM chembience/rdkit:0.4.0

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "user_client.request_base:app", "--host", "0.0.0.0", "--port", "80"]