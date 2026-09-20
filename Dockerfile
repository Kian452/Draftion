FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# lightgbm (the optional gradient-boosting backend) needs the OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY web ./web
COPY deploy ./deploy

# The app reads and writes everything under ./data (battles DB, model, roster, maps).
# A Railway volume mounted at /app/data hides whatever the image contains there, so
# the reference data and the demo model ship in deploy/seed and are copied in on
# start when missing. `cp -n` never overwrites: a model trained on the volume, or a
# refreshed roster, survives redeploys.
# src.serve runs the web app on $PORT (Railway sets it; 8000 locally) and, when BS_API_TOKEN is set,
# a background crawler + retrainer in the same process (see its docstring for the settings).
CMD ["sh", "-c", "mkdir -p data && cp -n deploy/seed/* data/ && exec python -m src.serve"]
