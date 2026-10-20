# Build from the repository root:  docker build -f api/Dockerfile -t papertrail-api .
FROM python:3.11-slim

# xgboost needs the GNU OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/
COPY data/demo/ data/demo/
ENV PAPERTRAIL_DEMO_DIR=/app/data/demo

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
