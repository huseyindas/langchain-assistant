FROM python:3.10-slim

RUN apt-get update && apt-get -y install \
    build-essential \
    libpq-dev \
    python3-dev \
    && apt-get clean

WORKDIR /backend

RUN pip install --upgrade pip && \
    pip install pipenv

COPY Pipfile Pipfile.lock /backend/

RUN pipenv install --deploy --ignore-pipfile

COPY . /backend

CMD ["pipenv", "run", "uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]
