FROM python:3.10-slim
 
WORKDIR /app
 
COPY requirements.txt .
 
# --no-install-recommends is REQUIRED on every apt-get line
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6       \
    libxext6     && \
    rm -rf /var/lib/apt/lists/*
 
# All packages pinned with ==X.Y.Z
RUN pip install --no-cache-dir \
    fastapi==0.111.0                    \
    uvicorn==0.29.0                     \
    ultralytics==8.2.0                  \
    Pillow==10.3.0                      \
    python-multipart==0.0.9             \
    numpy==1.26.4                       \
    opencv-python-headless==4.9.0.80    \
    pytest==8.2.0                       \
    httpx==0.27.0                       \
    pyyaml==6.0.1
 
COPY . .
RUN pip install --no-cache-dir -e .
 
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
