# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chinese education credential similarity detection system (学历证件相似度检测系统). Detects duplicate/fraudulent certificates using vector similarity search with Milvus, CLIP embeddings, and InsightFace facial recognition.

## Commands

### Running the Application

**Docker Compose (full stack):**
```bash
docker-compose up -d
# API available at http://localhost:8000/docs
```

**Local development (Milvus via Docker, app locally):**
```bash
# Start Milvus dependencies
docker-compose up -d milvus etcd minio

# Run app with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Environment variables:**
```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
DBSCAN_EPS=0.085           # image/face clustering radius
TEMPLATE_DBSCAN_EPS=0.09   # template clustering radius
```

## Architecture

### Three Milvus Collections (linked by `entity_id`)

| Collection | Vector | Dim | Purpose |
|---|---|---|---|
| `education_certificate_image` | CLIP | 512 | Full-image appearance similarity |
| `education_certificate_face` | InsightFace | 512 | Same-person detection |
| `education_certificate_template` | CLIP+HSV | 576 | Template/style clustering |

The 3-collection design works around a Milvus limitation (single vector field per collection in older versions).

### Template Vector Extraction (`vector_service.py`)

The most complex pipeline — designed to cluster certificates by template while ignoring personal details:
1. Detect faces → mask with gray (128,128,128)
2. Mask inner 20% region (name/date area) with gray
3. Apply Gaussian blur (radius=2) to reduce scan noise
4. Extract CLIP features from masked image → 512 dims
5. Extract HSV histogram from border region (outer 13%) → 64 dims
6. Concatenate `[CLIP(512) || HSV×1.5(64)]`, then L2-normalize → 576 dims

HSV×1.5 weighting was added to reach 100% accuracy on the test set (vs. 80% CLIP-only).

### Clustering Algorithms

- **Image/Face**: DBSCAN (`eps=0.085`, cosine distance, `min_samples=2`)
- **Template**: Agglomerative Clustering with complete linkage (`distance_threshold=0.09`)

Agglomerative clustering is used for templates specifically to avoid the DBSCAN "chaining" effect where intermediate images bridge dissimilar templates.

### Data Persistence

- Images: `app/uploads/{entity_id}.{ext}`
- ID-to-filename mapping: `data/id_to_filename.json` (managed by `id_filename_store.py`)
- Milvus data: Docker volume `milvus_data`

### Key Files

| File | Role |
|---|---|
| `app/main.py` | FastAPI routes: `/upload`, `/search`, `/cluster`, `/search_by_image`, `/upload_dir` |
| `app/vector_service.py` | `VectorService`: CLIP + InsightFace + HSV extraction |
| `app/milvus_client.py` | `MilvusClient`: Milvus connection, insert, search, get_all_vectors |
| `app/cluster_service.py` | `ClusterService`: DBSCAN and Agglomerative clustering |
| `app/config.py` | `Settings`: all tunable thresholds and dimensions |
| `app/schemas.py` | Pydantic request/response models |
| `static/groups.html` | Web UI with lazy-loaded image groups |

### Models (auto-downloaded on first run, ~1-2GB)

- CLIP: `openai/clip-vit-base-patch32` (cached in `~/.cache/huggingface`)
- InsightFace: `buffalo_l` model

## API Quick Reference

```bash
POST /upload              # Multipart: upload certificate images
POST /upload_dir          # ?directory=/path - batch import from server path
POST /search              # Body: {id, vector_field, top_k} or {*_vector, vector_field, top_k}
POST /search_by_image     # Form: file + vector_field + top_k
GET  /cluster             # ?vector_field=template_vector - cluster all data
GET  /view                # Web UI for visualizing clusters
GET  /api/image/{id}      # Retrieve image by entity_id
GET  /health              # Health check + collection stats
```

## Test Data

`docs/expected_similar_groups.md` documents 14 test images (1.png–14.png) with expected groupings (Groups A–E). Use this to validate clustering parameter changes.
