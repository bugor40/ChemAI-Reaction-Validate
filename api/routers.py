from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from api.celery_task.task import get_probability_task
from api.config.celery_utils import get_task_info

router = APIRouter(prefix='/predict', tags=['Proba'], responses={404: {"description": "Not found"}})
templates = Jinja2Templates(directory="./")

@router.get("/")
def root():
    return FileResponse("api/index/index.html")

@router.post("/result")
def postdata(request: Request, reactive: list = Form(), 
             product=Form()):
     task_id = get_probability_task.delay(reactive, product).id
     return f"Модель обучается, id модели {task_id}"

    # proba = get_probability_task(reactive, product)
    # return templates.TemplateResponse("api/index/index_output.html", {"request": request, 
    #                                                                     "reaction": f"{'+'.join(reactive)} = {product}",
    #                                                                     "proba": proba})

@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    return get_task_info(task_id)