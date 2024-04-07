import uvicorn as uvicorn
from fastapi import FastAPI

from api.config.celery_utils import create_celery
from api.routers import router


def create_app() -> FastAPI:
    current_app = FastAPI()

    current_app.celery_app = create_celery()
    current_app.include_router(router)
    return current_app


app = create_app()
celery = app.celery_app


# if __name__ == "__main__":
#     uvicorn.run("main:app", port=8000, reload=True)

# pip install -r requirements.txt
# pip freeze requirements.txt
# brew install rabbitmq
# start rabbitmq
# https://docs.celeryq.dev/en/stable/userguide/configuration.html#std-setting-worker_prefetch_multiplier