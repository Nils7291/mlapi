FROM python:3.10-slim-buster
WORKDIR /app
COPY app/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app/ /app
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]