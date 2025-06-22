FROM python:3.12-slim
RUN pip install pipenv
WORKDIR /app
COPY ["model.bin", "Pipfile", "Pipfile.lock", "starter.py", "./"]
RUN pipenv install --deploy --ignore-pipfile

ENTRYPOINT ["pipenv", "run", "python","starter.py"]
CMD []
