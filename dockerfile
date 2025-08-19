# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps (often optional for scikit-learn wheels, but safe)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# App Runner listens on 8080
ENV PORT=8080
EXPOSE 8080

# Use gunicorn to run Flask app (module:function)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
