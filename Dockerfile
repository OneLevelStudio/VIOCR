FROM python:3.10.10-slim-bullseye
ENV OS_ENV_DOCKER=1

WORKDIR /workdir
COPY . /workdir

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python3", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6969"]