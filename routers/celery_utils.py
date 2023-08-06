from celery import current_app as current_celery_app
from celery import Celery
# from env.config import 

def create_celery():
    celery_app = current_celery_app
    celery_app.conf.update(
        broker_url="redis://127.0.0.1:6379/0",
        result_backend="redis://127.0.0.1:6379/0"
    )

    return celery_app

