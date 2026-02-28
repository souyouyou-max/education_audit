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
   │ DINOv2/CLIP │ │ InsightFace │ │  Cluster    │
   │ (图像+模板)  │ │  (人脸)     │ │  Service    │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          └───────┬───────┘               │
                  ▼                       ▼
          ┌─────────────┐         ┌─────────────┐
          │   Milvus     │◄────────│  5 阶段管道  │
          │ (向量存储/检索)│         │  (聚类分析)  │
          └──────┬──────┘         └─────────────┘
                 │
       ┌─────┬──┴──┬─────┐
       ▼     ▼     ▼     ▼
     etcd  MinIO  HNSW  Collection
     (元数据)(对象) (索引) ×3
```

## 核心能力

| 能力 | 说明 | 模型 | 向量维度 |
|------|------|------|---------|
| **模板聚类** | 按证书模板分组（边框、背景、布局），忽略姓名/照片等个性化内容 | DINOv2-large | 1024 维 |
| **人脸比对** | 检测同一人脸出现在多张证书中 | InsightFace (buffalo_l) | 512 维 |
| **整体相似度** | 全图相似度检索，发现高度雷同的证件 | CLIP (ViT-B/32) | 512 维 |

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Web 框架 | FastAPI | 异步高性能，自带 OpenAPI 文档 |
| 模板特征 | DINOv2 (dinov2-large) | Meta 自监督视觉 Transformer，1024 维纯视觉结构特征 |
| 图像特征 | CLIP (clip-vit-base-patch32) | OpenAI 视觉语言模型，512 维 |
| 人脸识别 | InsightFace (buffalo_l) | 人脸检测 + 512 维嵌入向量 |
| 向量数据库 | Milvus 2.x | HNSW 索引，大规模余弦相似度检索 |
| 聚类算法 | HDBSCAN + KMeans + Union-Find | 5 阶段聚类管道（见下文） |
| 容器化 | Docker Compose | 一键部署 Milvus + etcd + MinIO + App |

## 项目结构

```
education_audit/
├── app/
│   ├── main.py                    # FastAPI 路由（上传/搜索/聚类/目录导入）
│   ├── config.py                  # 全局配置（维度、阈值、聚类参数）
│   ├── milvus_client.py           # Milvus 客户端（多集合管理、维度自动校验）
│   ├── vector_service.py          # 向量提取（CLIP + DINOv2 + InsightFace + HSV）
│   ├── cluster_service.py         # 聚类服务（5 阶段模板聚类 + image/face 聚类）
│   ├── cluster_images_dinov2.py   # 独立聚类脚本（调试/离线使用，与 API 逻辑对齐）
│   ├── id_filename_store.py       # ID-文件名映射（JSON 持久化）
│   ├── schemas.py                 # Pydantic 数据模型
│   └── utils.py                   # 图片验证、加载工具函数
├── static/
│   └── groups.html                # Web UI（聚类结果可视化，懒加载图片分组）
├── data/
│   └── id_to_filename.json        # ID-文件名映射存储
├── docs/
│   └── expected_similar_groups.md  # 测试数据标注（14 张图片的预期分组）
├── uploads/                       # 上传的证书图片（entity_id.ext）
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

### Milvus 集合设计

由于 Milvus 旧版不支持单集合多向量字段，拆分为三个集合，通过 `entity_id` 关联：

| 集合名 | 向量字段 | 维度 | 模型 | 用途 |
|--------|---------|------|------|------|
| `education_certificate_image` | `image_vector` | 512 | CLIP | 全图相似度检索 |
| `education_certificate_face` | `face_vector` | 512 | InsightFace | 人脸比对 |
| `education_certificate_template` | `template_vector` | 1088 | DINOv2+HSV | 模板风格搜索 |

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
export MILVUS_HOST=localhost              # Milvus 地址，默认 localhost
export MILVUS_PORT=19530                  # Milvus 端口，默认 19530
export DBSCAN_EPS=0.085                   # image/face 聚类半径
export TEMPLATE_HDBSCAN_EPSILON=0.12      # 模板聚类 HDBSCAN epsilon
export TEMPLATE_KMEANS_N=5                # KMeans 预分组数
export TEMPLATE_MERGE_THRESHOLD=0.37      # 跨组合并+噪声回收阈值
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

### 5. 其他接口

```bash
# Web UI 可视化聚类结果
curl http://localhost:8001/view

# 获取图片
curl http://localhost:8001/api/image/{entity_id}

# 健康检查 + 集合统计
curl http://localhost:8001/health
```

## 模板聚类原理

模板聚类是本系统的核心功能，目标是将使用相同模板的证书聚为一组。采用 **5 阶段管道** 设计，每个阶段解决一类特定问题。

### 管道总览

```
图片 → ① KMeans颜色预分组 → ② HDBSCAN精细聚类 → ③ 布局分裂 → ④ 跨组合并 → ⑤ 噪声回收 → 分组结果
```

| 阶段 | 算法 | 输入特征 | 解决的问题 |
|------|------|---------|-----------|
| ① 颜色预分组 | KMeans (K=5) | BGR均值 + 红印比例 (4D) | 粗分颜色差异大的证书，减少搜索空间 |
| ② 精细聚类 | HDBSCAN (leaf) | DINOv2 CLS token (1024D) | 同预组内按视觉结构精细聚类 |
| ③ 布局分裂 | Canny 边缘不对称度 | 灰度图左/右边缘密度 | 分离单页 vs 对折双页（DINOv2 无法区分） |
| ④ 跨组合并 | Union-Find + 布局兼容 | 簇质心欧氏距离 | 重新连接被 KMeans 错误分开的同模板 |
| ⑤ 噪声回收 | 最近质心匹配 | 噪声点到簇质心距离 | 挽救被孤立到错误预组的图片 |

### 第一阶段：KMeans 颜色预分组

提取 4 维颜色特征，用 KMeans 粗分为 5 组：

```
特征 = [B均值/255, G均值/255, R均值/255, 红印比例×100/10]
```

- **BGR 均值**：去除极亮/极暗像素（中值滤波 + 掩码 [20,235]）后的平均颜色
- **红印比例**：HSV 色彩空间中检测红色像素（H∈[0,10]∪[170,180], S≥70, V≥50），乘以 100/10 使其与 BGR 同量级

### 第二阶段：HDBSCAN 精细聚类

在每个颜色预组内，用 DINOv2-large 提取 1024 维视觉结构特征，L2 归一化后进行 HDBSCAN 聚类。

**为什么用 DINOv2 而不是 CLIP？**
- CLIP 是图文对比学习，特征偏"语义"——文字内容会影响特征
- DINOv2 是纯视觉自监督，特征偏"结构/纹理"——更关注排版、色块布局

**为什么用 `leaf` 而不是 `eom`？**

| 策略 | 原理 | 倾向 |
|------|------|------|
| eom (Excess of Mass) | 选择层级树中"持久性"最高的簇 | 少量大簇（易过度合并） |
| **leaf** | 选择层级树的叶节点 | 多量小簇（更细粒度） |

实测中 `eom` 会将 16 张不同模板的证书合成 1 个巨型簇，`leaf` 能正确拆为 3 个子组。

### 第三阶段：布局分裂

DINOv2 无法区分"单页证书"和"对折双页证书"（特征距离完全重叠），用 Canny 边缘检测的**左右不对称度**解决：

```python
edges = cv2.Canny(gray, 50, 150)
left_density  = edges[:, :w//2].mean()
right_density = edges[:, w//2:].mean()
asymmetry = |left - right| / (left + right)
```

| 类型 | 不对称度 | 原因 |
|------|---------|------|
| 单页证书 | < 0.04 | 左右内容对称 |
| 对折双页 | > 0.06 | 左=封面，右=内容，边缘密度差异大 |

阈值 **0.05** 位于间隔中间，100% 准确分离。

**关键设计**：布局分裂在跨组合并**之前**执行。如果先合并再分裂，混合布局簇的质心会被"平均掉"，导致后续错误合并。

### 第四阶段：跨组合并

KMeans 按颜色分组时，同模板但色调不同（扫描亮度差异等）可能被分到不同预组。用 Union-Find（并查集）合并质心距离 < 0.37 的簇：

```
对每对簇 (A, B):
  if 质心欧氏距离 < 0.37 AND 布局类型相同:
    union(A, B)
```

**布局兼容性约束**：只合并布局类型相同的簇（都是单页或都是对折），防止单页证书与对折证书因 DINOv2 距离接近而错误合并。

**Union-Find** 支持传递性合并（A-B 合并 + B-C 合并 → A-B-C 同组），路径压缩优化后近似 O(1) 时间复杂度。

### 第五阶段：噪声回收

被 KMeans 分到缺乏同类的预组中的图片，在 HDBSCAN 中会成为噪声点。将每个噪声点分配到最近的簇质心（距离 < 0.37）：

```
对每个噪声点 p:
  找距离 p 最近的簇质心 C
  if dist(p, C) < 0.37:
    将 p 归入该簇
```

### Milvus 存储的模板向量

除了聚类用的纯 DINOv2 特征，系统还在 Milvus 中存储用于**搜索**的混合模板向量（1088 维 = DINOv2 1024 + HSV 64）：

```
原始证书图像
    ├─ 1. InsightFace 检测人脸 → 灰色遮罩
    ├─ 2. 遮罩内部区域 → 去除姓名/日期
    ├─ 3. 高斯模糊(radius=2) → 消除扫描噪声
    ├─ 4. DINOv2 提取 1024 维结构特征
    ├─ 5. HSV 直方图提取 64 维颜色特征（边框区域）
    └─ 6. 拼接 [DINOv2(1024) + HSV×1.5(64)] → L2归一化 → 1088 维
```

聚类接口直接从原图提取纯 DINOv2 特征进行 5 阶段管道聚类，不使用 Milvus 中的混合向量，避免 HSV 混合干扰聚类精度。

### 独立聚类脚本

`app/cluster_images_dinov2.py` 是与 API 逻辑完全对齐的独立脚本，可用于离线调试：

```bash
python app/cluster_images_dinov2.py \
  --image_dir /path/to/images \
  --output_dir /path/to/output \
  --fine_epsilon 0.12 \
  --merge_threshold 0.37
```

输出按模板分组的文件夹结构，方便人工校验。

## 调优指南

### 核心参数

| 参数 | 默认值 | 环境变量 | 作用 | 调优方向 |
|------|--------|---------|------|---------|
| HDBSCAN epsilon | 0.12 | `TEMPLATE_HDBSCAN_EPSILON` | 精细聚类粒度 | 调大→更多合并；调小→更多分裂 |
| KMeans N | 5 | `TEMPLATE_KMEANS_N` | 颜色预分组数 | 调大→预组更细；调小→预组更粗 |
| 合并阈值 | 0.37 | `TEMPLATE_MERGE_THRESHOLD` | 跨组合并+噪声回收 | 调大→更多合并；调小→更保守 |
| 布局分裂阈值 | 0.05 | 代码内硬编码 | 单页/对折分离 | 一般无需调整 |
| DBSCAN eps | 0.085 | `DBSCAN_EPS` | image/face 聚类半径 | 同上 |

### 调优方法

```bash
# 通过环境变量调整，无需改代码
export TEMPLATE_HDBSCAN_EPSILON=0.10   # 更严格的精细聚类
export TEMPLATE_MERGE_THRESHOLD=0.35   # 更保守的跨组合并
```

**判断标准**：
- 同模板证书被分成多个组 → 调大 `TEMPLATE_MERGE_THRESHOLD` 或 `TEMPLATE_HDBSCAN_EPSILON`
- 不同模板证书被合到一组 → 调小对应参数
- 单页/对折混在一组 → 检查布局分裂阈值（通常不需要调）

### 距离参考值

基于 61 张真实证书测试得出的 DINOv2 欧氏距离参考：

| 场景 | 距离范围 | 说明 |
|------|---------|------|
| 同模板（色调相近） | 0.28 - 0.35 | 同一模板、相近扫描条件 |
| 同模板（色调差异） | 0.35 - 0.40 | 同一模板、扫描亮度/色温差异 |
| 不同模板 | 0.48+ | 不同证书模板 |

## 注意事项

1. **首次运行**：DINOv2、CLIP 和 InsightFace 模型会自动下载（约 2-3GB），需要网络连接
2. **GPU 加速**：如有 NVIDIA GPU，PyTorch 和 ONNX Runtime 会自动使用 CUDA
3. **内存需求**：建议至少 8GB，模型加载约需 3-4GB
4. **维度变更**：修改向量维度后，Milvus 集合会自动检测并重建（旧数据丢失，需重新导入）
5. **数据持久化**：上传图片存储在 `uploads/`，ID 映射存储在 `data/id_to_filename.json`，Milvus 数据通过 Docker Volume 持久化
6. **聚类一致性**：API 聚类接口与独立脚本 `cluster_images_dinov2.py` 逻辑完全对齐，可互相验证

## License

MIT License
