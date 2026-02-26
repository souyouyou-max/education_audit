# 学历证件相似度检测系统

基于 Milvus 向量数据库和深度学习模型的学历证件智能稽核系统，支持证书模板聚类、人脸比对和相似度检索。

## 系统架构

```
                    ┌─────────────┐
                    │   客户端     │
                    │ (curl/浏览器) │
                    └──────┬──────┘
                           │ HTTP
                    ┌──────▼──────┐
                    │   FastAPI    │
                    │  Web Server  │
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┤   │   ├────────────┐
          ▼            ▼   │   ▼            ▼
   ┌─────────────┐ ┌──────▼──────┐ ┌─────────────┐
   │ CLIP Model  │ │ InsightFace │ │  Cluster    │
   │ (图像+模板)  │ │  (人脸)     │ │  Service    │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          └───────┬───────┘               │
                  ▼                       ▼
          ┌─────────────┐         ┌─────────────┐
          │   Milvus     │◄────────│  DBSCAN     │
          │ (向量存储/检索)│         │  (聚类分析)  │
          └──────┬──────┘         └─────────────┘
                 │
       ┌─────┬──┴──┬─────┐
       ▼     ▼     ▼     ▼
     etcd  MinIO  HNSW  Collection
     (元数据)(对象) (索引) ×3
```

## 核心能力

| 能力 | 说明 | 向量维度 |
|------|------|---------|
| **模板聚类** | 按证书模板分组（边框、背景、布局），忽略姓名/照片等个性化内容 | 576 维 (CLIP 512 + HSV 64) |
| **人脸比对** | 检测同一人脸出现在多张证书中 | 512 维 (InsightFace) |
| **整体相似度** | 全图相似度检索，发现高度雷同的证件 | 512 维 (CLIP) |

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步高性能，自带 OpenAPI 文档 |
| 图像特征 | CLIP (clip-vit-base-patch32) | OpenAI 预训练视觉语言模型，512 维输出 |
| 人脸识别 | InsightFace (buffalo_l) | 人脸检测 + 512 维人脸嵌入向量 |
| 向量数据库 | Milvus 2.x | HNSW 索引，支持大规模余弦相似度检索 |
| 聚类算法 | DBSCAN | 无需预设簇数，自动发现密度聚类 |
| 容器化 | Docker Compose | 一键部署 Milvus + etcd + MinIO + App |

## 项目结构

```
education_audit/
├── app/
│   ├── main.py              # FastAPI 路由（上传/搜索/聚类/目录导入）
│   ├── config.py            # 全局配置（维度、阈值、DBSCAN 参数）
│   ├── milvus_client.py     # Milvus 客户端（多集合管理、维度自动校验）
│   ├── vector_service.py    # 向量提取（CLIP + InsightFace + HSV 直方图）
│   ├── cluster_service.py   # 聚类服务（image/face/template 三种聚类）
│   ├── id_filename_store.py # ID-文件名映射（JSON 持久化）
│   ├── schemas.py           # Pydantic 数据模型
│   └── utils.py             # 图片验证、加载工具函数
├── data/
│   └── id_to_filename.json  # ID-文件名映射存储
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

### Milvus 集合设计

由于 Milvus 旧版不支持单集合多向量字段，拆分为三个集合，通过 `entity_id` 关联：

| 集合名 | 向量字段 | 维度 | 用途 |
|--------|---------|------|------|
| `education_certificate_image` | `image_vector` | 512 | 全图 CLIP 特征 |
| `education_certificate_face` | `face_vector` | 512 | 人脸嵌入向量 |
| `education_certificate_template` | `template_vector` | 576 | 模板特征（CLIP+HSV 混合） |

## 快速开始

### 方式一：Docker Compose

```bash
# 启动所有服务（Milvus + etcd + MinIO + App）
docker-compose up -d

# 查看日志
docker-compose logs -f app
```

### 方式二：本地运行

```bash
# 1. 启动 Milvus 依赖
docker-compose up -d milvus etcd minio

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 启动应用
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

环境变量（可选）：

```bash
export MILVUS_HOST=localhost     # Milvus 地址，默认 localhost
export MILVUS_PORT=19530         # Milvus 端口，默认 19530
export DBSCAN_EPS=0.085          # 通用聚类 eps
export TEMPLATE_DBSCAN_EPS=0.11  # 模板聚类 eps
```

## API 接口

启动后访问 `http://localhost:8001/docs` 查看交互式 API 文档。

### 1. 上传图片

```bash
# 单张/多张上传
curl -X POST http://localhost:8001/upload \
  -F "files=@1.png" -F "files=@2.png" -F "files=@3.png"
```

### 2. 从目录批量导入

```bash
# 导入服务器本地目录下所有图片
curl -X POST "http://localhost:8001/upload_dir?directory=/path/to/images"
```

### 3. 模板聚类

```bash
# 按证书模板分组（推荐用于模板稽核）
curl http://localhost:8001/cluster?vector_field=template_vector

# 按全图相似度分组
curl http://localhost:8001/cluster?vector_field=image_vector

# 按人脸分组
curl http://localhost:8001/cluster?vector_field=face_vector
```

### 4. 相似度搜索

```bash
# 根据已有记录 ID 搜索相似证件
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"id": 123, "vector_field": "template_vector", "top_k": 5}'

# 上传图片直接搜索
curl -X POST http://localhost:8001/search_by_image \
  -F "file=@query.png" -F "vector_field=template_vector" -F "top_k=5"
```

### 5. 健康检查

```bash
curl http://localhost:8001/health
```

## 模板聚类原理

模板聚类是本系统的核心功能，目标是将使用相同模板的证书聚为一组。

### 特征提取流程

```
原始证书图像
    │
    ├─ 1. InsightFace 检测人脸 → 用灰色(128,128,128)遮罩人脸区域
    │
    ├─ 2. 遮罩内部 56% 区域 → 用灰色填充（去除姓名/日期等文字）
    │
    ├─ 3. 轻微高斯模糊(radius=2) → 消除扫描噪声
    │
    ├─ 4. CLIP 提取 512 维结构特征 → 捕获边框图案、布局结构
    │
    ├─ 5. HSV 直方图提取 64 维颜色特征 → 捕获边框颜色（红/绿/金等）
    │
    └─ 6. 拼接 [CLIP(512) + HSV×2.0(64)] → L2归一化 → 576 维模板向量
```

### 为什么需要 HSV 直方图？

经过对 14 张真实证书的测试验证：

| 方案 | 正确分组率 | 组内外距离间隔 |
|------|-----------|---------------|
| 纯 CLIP（遮罩） | 4/5 (80%) | -0.024 (重叠) |
| 纯 HSV 直方图 | 5/5 (100%) | +0.208 |
| **CLIP + HSV 混合** | **5/5 (100%)** | **+0.171** |

CLIP 擅长结构/语义理解，但对边框颜色这种低级视觉差异不够敏感。HSV 直方图直接捕获颜色分布，两者互补。

## 调优指南

### 1. 模板聚类 eps 调优

`TEMPLATE_DBSCAN_EPS` 是模板聚类最关键的参数，控制"多近算同一组"：

| eps 值 | 效果 | 适用场景 |
|--------|------|---------|
| 0.05 | 非常严格，只有极相似的才分到一组 | 模板差异非常细微时 |
| **0.11** | **默认值，平衡精度和召回** | **大多数场景** |
| 0.15 | 较宽松，更多证书会被归到同一组 | 扫描质量差异大时 |
| 0.20 | 很宽松，不同模板可能被合并 | 不推荐 |

**调优方法**：

```bash
# 通过环境变量调整，无需改代码
export TEMPLATE_DBSCAN_EPS=0.08
```

**判断标准**：
- 如果同一模板的证书被分成了多个组 → **调大 eps**
- 如果不同模板的证书被合并到一组 → **调小 eps**

### 2. 通用聚类 eps 调优

`DBSCAN_EPS`（默认 0.085）用于 `image_vector` 和 `face_vector` 聚类：

```bash
export DBSCAN_EPS=0.085  # 对应余弦相似度约 0.915
```

### 3. 边框区域比例

`TEMPLATE_BORDER_RATIO`（默认 0.15）控制 HSV 直方图取样的边框宽度：

- 较大值（0.20）：采样更多背景区域，对背景纹理差异更敏感
- 较小值（0.10）：只采样最外圈边框，对边框线条颜色更敏感

### 4. HSV 权重

在 `vector_service.py` 中 HSV 特征乘以 2.0 的权重：

```python
combined = np.concatenate([clip_vec, hsv_vec * 2.0])
```

- 增大权重（如 3.0）：更依赖颜色差异，适合边框颜色差异明显的场景
- 减小权重（如 1.0）：更依赖 CLIP 结构特征，适合颜色相近但布局不同的场景

### 5. 模型升级

当前使用 `clip-vit-base-patch32`（512 维），可升级为更强模型：

```python
# config.py
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # 768 维，更精确
```

注意升级后需同步修改 `IMAGE_VECTOR_DIM` 和 `TEMPLATE_VECTOR_DIM`，并重新导入所有图片。

### 6. 人脸检测灵敏度

```python
# vector_service.py extract_face_vector()
min_quality_score=0.5  # 降低可检测更多低质量人脸
face_size < 80         # 降低可检测更小的人脸
```

## 注意事项

1. **首次运行**：CLIP 和 InsightFace 模型会自动下载（约 1-2GB），需要网络连接
2. **GPU 加速**：如有 NVIDIA GPU，PyTorch 和 ONNX Runtime 会自动使用 CUDA
3. **内存需求**：建议至少 8GB，模型加载约需 2-3GB
4. **维度变更**：修改向量维度后，Milvus 集合会自动检测并重建（旧数据会丢失，需重新导入）
5. **数据持久化**：ID-文件名映射存储在 `data/id_to_filename.json`，Milvus 数据通过 Docker Volume 持久化

## License

MIT License
