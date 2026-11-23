FROM python:3.12-alpine

WORKDIR /app

COPY model/itemknn_v1 ./model
COPY src/app.py src/itemknn.py ./

ENV MODEL_DIR="/app/model"

RUN pip install --no-cache-dir flask numpy pandas scipy

EXPOSE 8000

CMD ["python3", "app.py"]
