from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import routers_main
import os

app = FastAPI()


os.makedirs("static/stacked", exist_ok=True)
os.makedirs("static/perturbations", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/adversarial", exist_ok=True)

#Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(routers_main.router)
