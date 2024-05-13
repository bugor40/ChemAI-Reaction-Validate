# pull official base image
FROM python:3.9.19-bullseye

COPY ./.aws/credentials /root/.aws/credentials
COPY ./.aws/config /root/.aws/config

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .