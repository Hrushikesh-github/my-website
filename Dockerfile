# pull official base image
FROM python:3.11.2-slim-buster

RUN pip install --upgrade pip
COPY ./requirements.txt .
# Install PyTorch and torchvision with cpu
RUN pip install -r requirements.txt
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# add app
COPY . .

# Specify the command to be executed when running the container
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["gunicorn -b 127.0.0.1:3100 -w 4 -k uvicorn.workers.UvicornWorker main:app"]
CMD ["gunicorn", "-b", "0.0.0.0:3400", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
# CMD ["gunicorn", "-b", "0.0.0.0:3400", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--proxy-headers", "main:app"]
# RUN THE BELOW COMMAND AFTER BUILDING
# docker run -d -p 3400:3400 --name my-container my-dock