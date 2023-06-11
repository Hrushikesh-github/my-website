from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import routers_main

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(routers_main.router)
