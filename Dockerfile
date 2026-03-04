FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（单层，利用缓存）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件（利用 Docker 层缓存：代码变更不重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制应用代码
COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/

# 创建运行时目录
RUN mkdir -p uploads data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
