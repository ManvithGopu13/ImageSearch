# 🔍 Image Search RAG System

An intelligent image search system powered by AI, combining OCR, object detection, image captioning, and semantic search with RAG (Retrieval-Augmented Generation).

## ✨ Features

- 📸 **Upload & Process Images** - Automatic analysis of uploaded images
- 🔍 **Semantic Search** - Natural language search across your image collection
- 🤖 **AI-Powered Analysis**:
  - OCR text extraction (EasyOCR)
  - Object detection (YOLOv8)
  - Image captioning (BLIP)
  - Vector embeddings (CLIP)
- 💾 **Vector Database** - ChromaDB for efficient similarity search
- 🧠 **RAG Integration** - Enhanced search with NVIDIA LLM
- 📊 **Interactive Visualizations** - 2D/3D network maps of image relationships
- 🎯 **Multi-Field Search** - Search by visual similarity, OCR text, captions, and detected objects

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 12.1+ (for GPU acceleration)
- NVIDIA API Key (optional, for enhanced LLM features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ManvithGopu13/ImageSearch.git
cd ImageSearch
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install PyTorch with CUDA**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure environment**
```bash
cp config.env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

6. **Run the server**
```bash
python run_server.py
```

7. **Open in browser**
```
http://localhost:8000
```

## 📁 Project Structure

```
ImageSearch/
├── backend/
│   ├── config.py              # Configuration settings
│   ├── image_processor.py     # Image analysis pipeline
│   ├── vector_database.py     # ChromaDB interface
│   ├── rag_search.py          # RAG search engine
│   ├── visualizer.py          # 2D/3D visualizations
│   ├── visualizer_3d.py       # Advanced 3D network maps
│   └── main.py                # FastAPI application
├── frontend/
│   └── index.html             # Web interface
├── config.env.example         # Environment template
├── requirements.txt           # Python dependencies
├── run_server.py              # Server launcher
└── test_system.py             # System tests
```

## 🎯 Usage

### 1. Upload Images
- Drag and drop or click to upload images
- System automatically extracts:
  - Text (OCR)
  - Objects
  - Captions
  - Vector embeddings

### 2. Search
- Enter natural language queries like:
  - "cat playing with ball"
  - "screenshot with ngrok"
  - "dashboard with charts"
- Toggle LLM enhancement for smarter results

### 3. View Results
- See top matching images with:
  - Similarity scores
  - Detected objects
  - Extracted text
  - AI captions

### 4. Visualize
- **Network Map** - 2D graph showing image relationships
- **3D Vector Space** - Interactive 3D scatter plot with clusters

## 🔧 Configuration

Edit `.env` file:

```env
# NVIDIA API
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL_NAME=meta/llama-3.1-8b-instruct

# Models
CLIP_MODEL=ViT-B/32
BLIP_MODEL=Salesforce/blip-image-captioning-base
YOLO_MODEL=yolov8n.pt

# Search Settings
TOP_K_RESULTS=10
SIMILARITY_THRESHOLD=0.22
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.9+
- **AI Models**:
  - EasyOCR - Text extraction
  - YOLOv8 - Object detection
  - BLIP - Image captioning
  - CLIP - Image embeddings
- **Vector DB**: ChromaDB
- **LLM**: NVIDIA AI Endpoints (LangChain)
- **Visualization**: Plotly, scikit-learn
- **Frontend**: HTML, JavaScript, CSS

## 📊 Search Features

### Dual-Embedding Blending
- Combines original and enhanced queries (70/30 ratio)
- Better semantic understanding

### Multi-Field Boosting
- Results boosted when keywords match in:
  - OCR text (+5%)
  - Captions (+3%)
  - Object labels (+4%)

### Smart Filtering
- Similarity threshold: 22%
- Configurable top-K results
- Keyword-aware ranking

## 🎨 Visualization

### Network Map
- 2D graph with colored clusters
- Connection lines show similarity
- Hover to preview images
- Pan, zoom, and explore

### 3D Vector Space
- Interactive 3D scatter plot
- Cluster identification
- Connection visualization
- Rotatable camera view

## 🧪 Testing

```bash
python test_system.py
```

## 📝 API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI CLIP
- Salesforce BLIP
- Ultralytics YOLOv8
- EasyOCR
- LangChain & NVIDIA AI

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Python, FastAPI, and cutting-edge AI models**
