from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import routers_main
# from routers.celery_utils import create_celery
import os
from routers.celery_utils import create_celery


app = FastAPI()

os.makedirs("static/stacked", exist_ok=True)
os.makedirs("static/perturbations", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/adversarial", exist_ok=True)

app = FastAPI()

celery_app = create_celery()
app.celery_app = celery_app
#Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(routers_main.router)
