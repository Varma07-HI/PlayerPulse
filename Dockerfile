FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY app ./app
COPY models ./models
ENV PORT=8000
CMD sh -c 'uvicorn app.main:app --host 0.0.0.0 --port ${PORT}'