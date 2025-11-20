# -------- BASE IMAGE --------
FROM python:3.11-slim

# -------- SYSTEM SETUP --------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-kan \
    libtesseract-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------- WORKDIR & DEPENDENCIES --------
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- COPY CODE --------
COPY backend /app/backend

# -------- VERIFY TESSERACT --------
RUN which tesseract && tesseract --version

# -------- EXPOSE PORT --------
EXPOSE 8080

# -------- START COMMAND --------
CMD ["uvicorn", "backend.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8080"]
