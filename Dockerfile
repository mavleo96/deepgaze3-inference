FROM bitnami/python:3.8

WORKDIR /app

ENV HTTP_PORT=4000

RUN apt-get update \
    && apt-get -y install gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY ./requirements-server.txt ./
RUN python -m pip install --no-cache -U pip \
    && python -m pip install --no-cache -r requirements-server.txt

COPY ./model_server.py ./

EXPOSE $HTTP_PORT

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "--pythonpath", ".", "--access-logfile", "-", "model_server:app"]