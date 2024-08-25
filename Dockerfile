FROM python:3.10-slim

WORKDIR /backend

RUN pip install --upgrade pip && \
    pip install pipenv

COPY Pipfile Pipfile.lock /backend/

RUN pipenv install --deploy --ignore-pipfile

COPY . /backend

CMD ["pipenv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
