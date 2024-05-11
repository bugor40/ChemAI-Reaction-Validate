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
def postdata(
    request: Request, 
    reactive: list = Form(), 
    product=Form()
):
     url = '158.160.60.224' #'chem-ai.ru' #'0.0.0.0:80'
     task_id = get_probability_task.delay(reactive, product).id
     return templates.TemplateResponse(
                "api/index/index_go.html",
                    {
                    "request": request,               
                    "task_id": task_id,
                    'url': url,
                    }
            )


@router.get("/task/{task_id}")
async def get_probability_status(
    request: Request, 
    task_id: str
):
    result = get_task_info(task_id)
    task_status = result['task_status']
    task_result = result['task_result']

    if task_status == 'SUCCESS':
        reaction = task_result['reaction']
        proba = task_result['proba']

        result = templates.TemplateResponse(
            "api/index/index_output_success.html", 
            {
                "request": request,
                "reaction": reaction,
                "proba": proba
            }
        )
    else:
        result = templates.TemplateResponse(
                "api/index/index_output_progres.html",
                {
                    "request": request
                }
            )
    
    return result