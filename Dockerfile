FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY planet_hunter/ planet_hunter/
COPY planet-hunter.service .

RUN mkdir -p plots

EXPOSE 8420

CMD ["uvicorn", "planet_hunter.main:app", "--host", "0.0.0.0", "--port", "8420"]
