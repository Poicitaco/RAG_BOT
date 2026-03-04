# 🏥 Pharmaceutical AI Assistant - Hệ thống AI Agent Đa phương tiện Hỗ trợ Dược phẩm

## 📋 Tổng quan

Hệ thống AI Agent thông minh hỗ trợ người dân tra cứu thông tin dược phẩm, kiểm tra tương tác thuốc, tư vấn liều lượng và cảnh báo an toàn. Hệ thống sử dụng công nghệ RAG (Retrieval-Augmented Generation) kết hợp với multi-modal AI để xử lý văn bản, hình ảnh và giọng nói.

## ✨ Tính năng chính

### 🤖 AI Agents

- **Drug Information Agent**: Cung cấp thông tin chi tiết về thuốc
- **Interaction Check Agent**: Kiểm tra tương tác giữa các loại thuốc
- **Dosage Advisor Agent**: Tư vấn liều lượng phù hợp
- **Safety Monitor Agent**: Giám sát và cảnh báo an toàn

### 🎯 Đa phương tiện

- **Text**: Chat tư vấn thông minh với ngữ cảnh
- **Image**: Nhận diện thuốc qua hình ảnh (OCR + Vision AI)
- **Voice**: Nhận diện giọng nói và phản hồi bằng giọng nói

### 🔍 RAG System

- Vector search với ChromaDB/FAISS
- Semantic search thông tin thuốc
- Context-aware responses
- Real-time knowledge retrieval

### 🛡️ An toàn & Tuân thủ

- Kiểm tra tương tác thuốc tự động
- Cảnh báo chống chỉ định
- Xác thực liều lượng
- Medical disclaimer

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│              (Streamlit / React / Mobile App)                │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
│          ┌──────────┬──────────┬──────────┬────────┐        │
│          │  Auth    │  Rate    │  Logging │  CORS  │        │
│          │ Middleware│ Limiting │ Monitor  │ Handler│        │
└──────────┴──────────┴──────────┴──────────┴────────┴────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Agent Orchestrator                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │  Drug Info   │  │ Interaction  │  │   Dosage     │     │
│   │    Agent     │  │ Check Agent  │  │Advisor Agent │     │
│   └──────────────┘  └──────────────┘  └──────────────┘     │
│   ┌──────────────┐  ┌──────────────┐                       │
│   │   Safety     │  │   General    │                       │
│   │Monitor Agent │  │ Consult Agent│                       │
│   └──────────────┘  └──────────────┘                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    RAG Engine                                │
│   ┌──────────────────────────────────────────────────┐      │
│   │  Query Processing → Embedding → Vector Search    │      │
│   │  → Context Retrieval → Response Generation       │      │
│   └──────────────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                Multi-modal Services                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │    Text      │  │    Image     │  │    Voice     │     │
│   │  Processing  │  │ Recognition  │  │  Processing  │     │
│   └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Data Layer                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│   │  PostgreSQL  │  │  ChromaDB    │  │    Redis     │     │
│   │  (Metadata)  │  │  (Vectors)   │  │   (Cache)    │     │
│   └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Cấu trúc dự án

```
RAG/
├── backend/                    # Backend application
│   ├── agents/                # AI agents implementation
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Base agent class
│   │   ├── drug_info_agent.py
│   │   ├── interaction_agent.py
│   │   ├── dosage_agent.py
│   │   ├── safety_agent.py
│   │   └── orchestrator.py   # Agent orchestration
│   ├── rag/                  # RAG system
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Embedding generation
│   │   ├── vector_store.py   # Vector database
│   │   ├── retriever.py      # Document retrieval
│   │   └── generator.py      # Response generation
│   ├── models/               # Data models
│   │   ├── __init__.py
│   │   ├── drug.py          # Drug models
│   │   ├── user.py          # User models
│   │   └── conversation.py  # Conversation models
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   ├── text_service.py
│   │   ├── image_service.py
│   │   ├── voice_service.py
│   │   └── safety_service.py
│   ├── api/                 # API endpoints
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── drug.py
│   │   │   ├── image.py
│   │   │   └── voice.py
│   │   └── dependencies.py
│   ├── utils/               # Utilities
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── validators.py
│   │   └── helpers.py
│   ├── config/              # Configuration
│   │   ├── __init__.py
│   │   └── settings.py
│   └── main.py              # Application entry point
├── data/
│   ├── drugs/               # Drug database (CSV, JSON)
│   ├── embeddings/          # Vector embeddings
│   └── knowledge/           # Knowledge base
├── frontend/                # Streamlit frontend
│   ├── app.py
│   ├── components/
│   └── utils/
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_test.ipynb
│   └── 03_agent_testing.ipynb
├── tests/                   # Unit tests
│   ├── test_agents/
│   ├── test_rag/
│   └── test_services/
├── .env.example            # Environment variables template
├── .gitignore
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker setup
└── README.md              # This file
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- PostgreSQL 14+
- Redis 7+
- 8GB RAM (khuyến nghị 16GB)

### Bước 1: Clone và setup môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Cấu hình môi trường

```bash
# Copy file .env.example sang .env
copy .env.example .env  # Windows
# hoặc
cp .env.example .env    # Linux/Mac

# Chỉnh sửa file .env với API keys của bạn
# - OPENAI_API_KEY
# - Database credentials
# - Redis configuration
```

### Bước 3: Khởi tạo database

```bash
# Tạo database
# PostgreSQL:
createdb pharma_ai

# Chạy migrations
alembic upgrade head

# Import dữ liệu mẫu (nếu có)
python scripts/import_drug_data.py
```

### Bước 4: Khởi tạo Vector Store

```bash
# Tạo embeddings từ dữ liệu
python scripts/create_embeddings.py

# Kiểm tra vector store
python scripts/test_vector_search.py
```

### Bước 5: Chạy ứng dụng

```bash
# Chạy backend API
uvicorn backend.main:app --reload --port 8000

# Chạy frontend (terminal mới)
streamlit run frontend/app.py
```

## 🎮 Sử dụng

### API Examples

#### 1. Chat với AI Agent

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "Paracetamol 500mg dùng như thế nào?",
        "session_id": "user123"
    }
)
print(response.json())
```

#### 2. Nhận diện thuốc qua hình ảnh

```python
files = {"file": open("drug_image.jpg", "rb")}
response = requests.post(
    "http://localhost:8000/api/v1/image/recognize",
    files=files
)
print(response.json())
```

#### 3. Kiểm tra tương tác thuốc

```python
response = requests.post(
    "http://localhost:8000/api/v1/drug/check-interaction",
    json={
        "drugs": ["Aspirin", "Warfarin", "Ibuprofen"]
    }
)
print(response.json())
```

## 🧪 Testing

```bash
# Chạy tất cả tests
pytest

# Test với coverage
pytest --cov=backend --cov-report=html

# Test specific module
pytest tests/test_agents/

# Test với verbose
pytest -v
```

## 📊 Monitoring

- **API Metrics**: http://localhost:8000/metrics (Prometheus)
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/health

## 🔧 Development

### Code Quality

```bash
# Format code
black backend/

# Lint code
flake8 backend/

# Type checking
mypy backend/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding New Agent

Xem hướng dẫn chi tiết tại [docs/adding_agent.md](docs/adding_agent.md)

## 🐳 Docker Deployment

```bash
# Build và chạy với docker-compose
docker-compose up -d

# Xem logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📈 Roadmap

### Phase 1: MVP (Hiện tại)

- ✅ Cấu trúc dự án
- ⏳ RAG system cơ bản
- ⏳ Text chat agent
- ⏳ Streamlit UI

### Phase 2: Multi-modal

- ⏳ Image recognition
- ⏳ Voice input/output
- ⏳ Mobile responsive UI

### Phase 3: Advanced Features

- ⏳ Multi-agent collaboration
- ⏳ Personalized recommendations
- ⏳ Medical record integration
- ⏳ Prescription analysis

### Phase 4: Scale & Optimize

- ⏳ Performance optimization
- ⏳ Load balancing
- ⏳ Mobile app (Flutter/React Native)
- ⏳ Multi-language support

## ⚠️ Disclaimer

**CẢNH BÁO QUAN TRỌNG**: Hệ thống này chỉ mang tính chất tham khảo và không thay thế cho tư vấn y tế chuyên nghiệp. Luôn tham khảo ý kiến bác sĩ hoặc dược sĩ trước khi sử dụng bất kỳ loại thuốc nào.

## 📝 License

MIT License - Xem file LICENSE để biết chi tiết

## 👥 Contributors

- Bạn - Founder & Lead Developer

## 📞 Liên hệ

- Email: support@pharma-ai.com
- GitHub Issues: [github.com/yourrepo/issues](https://github.com/yourrepo/issues)

## 🙏 Acknowledgments

- LangChain team
- OpenAI
- Anthropic
- Open-source community
