from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import routers_main
import os
from fastapi import Request

app = FastAPI()

os.makedirs("static/stacked", exist_ok=True)
os.makedirs("static/perturbations", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/adversarial", exist_ok=True)

#Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.middleware("http")
async def middleware_events(request: Request, call_next):
    if os.getenv("FASTAPI_ENV") != "local":
        request.scope["scheme"] = "https"
        
    response = await call_next(request)

    return response

app.include_router(routers_main.router)
