# using Python 3.9 for Linux to keep the test image lightweight
FROM python:3.9-slim
WORKDIR /app
COPY requirements.gpu.txt requirements.txt
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]