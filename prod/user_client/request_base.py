from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
# import time


app = FastAPI()
templates = Jinja2Templates(directory="./")

@app.get("/")
def root():
    return FileResponse("user_client/index.html")
 
 
@app.post("/result")
def postdata(request: Request, reactive: list = Form(), 
             product=Form()):
    return templates.TemplateResponse("user_client/index_output.html", {"request": request, "reaction": f"{'+'.join(reactive)} = {product}"})
