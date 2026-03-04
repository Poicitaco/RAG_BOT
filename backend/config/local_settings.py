"""
Cấu hình cho Local AI Models
Thay thế OpenAI API bằng các models chạy local
"""
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class LocalAISettings(BaseSettings):
    """Cấu hình cho Local AI"""
    
    # ============= Chế độ AI =============
    # "local" hoặc "openai"
    AI_MODE: Literal["local", "openai"] = Field(
        default="local",
        description="Chế độ AI: local (chạy local) hoặc openai (dùng API)"
    )
    
    # ============= LOCAL LLM (Ollama) =============
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="URL của Ollama server"
    )
    OLLAMA_MODEL: str = Field(
        default="mistral:7b-instruct",
        description="Tên model Ollama (mistral:7b-instruct, llama3:8b, qwen2.5:7b, v.v.)"
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Model embedding của Ollama"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=120,
        description="Timeout cho Ollama requests (seconds)"
    )
    OLLAMA_TEMPERATURE: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature cho generation (0.0-2.0)"
    )
    OLLAMA_MAX_TOKENS: int = Field(
        default=2048,
        ge=100,
        le=32000,
        description="Số token tối đa cho mỗi response"
    )
    
    # ============= LOCAL EMBEDDINGS =============
    # Dùng sentence-transformers thay vì OpenAI embeddings
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="keepitreal/vietnamese-sbert",
        description="Model embedding local (hỗ trợ tiếng Việt tốt)"
    )
    # Các model tiếng Việt khác tốt:
    # - "keepitreal/vietnamese-sbert" (tốt nhất cho tiếng Việt)
    # - "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Số chiều của embedding vectors"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32,
        description="Batch size khi tạo embeddings"
    )
    EMBEDDING_DEVICE: str = Field(
        default="cuda",
        description="Device để chạy embedding model: 'cuda', 'cpu', 'mps'"
    )
    
    # ============= LOCAL WHISPER =============
    # Dùng faster-whisper thay vì OpenAI Whisper API
    WHISPER_MODEL_SIZE: str = Field(
        default="base",
        description="Kích thước model Whisper: tiny, base, small, medium, large-v3"
    )
    WHISPER_DEVICE: str = Field(
        default="cuda",
        description="Device cho Whisper: cuda hoặc cpu"
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="float16",
        description="Compute type: float16, int8, int8_float16"
    )
    WHISPER_LANGUAGE: str = Field(
        default="vi",
        description="Ngôn ngữ: vi (tiếng Việt) hoặc en"
    )
    
    # ============= PERFORMANCE TUNING =============
    USE_GPU: bool = Field(
        default=True,
        description="Sử dụng GPU nếu có"
    )
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=5,
        description="Số request đồng thời tối đa"
    )
    MODEL_CACHE_DIR: str = Field(
        default="./models_cache",
        description="Thư mục cache cho models"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
local_ai_settings = LocalAISettings()


# ============= RECOMMENDED MODELS =============
"""
KHUYẾN NGHỊ MODELS CHO HỆ THỐNG DƯỢC PHẨM:

1. LLM (Ollama):
   - mistral:7b-instruct (7GB) - Cân bằng tốt, tiếng Việt khá
   - qwen2.5:7b (7GB) - Rất tốt cho tiếng Việt
   - gemma2:9b (9GB) - Nhanh, chất lượng tốt
   - llama3.1:8b (8GB) - Tốt all-around

2. Embeddings:
   - keepitreal/vietnamese-sbert (300MB) - Tốt nhất cho tiếng Việt
   - VoVanPhuc/sup-SimCSE-VietNamese-phobert-base (400MB)
   - paraphrase-multilingual-MiniLM-L12-v2 (470MB)

3. Whisper:
   - base (140MB) - Cân bằng speed/accuracy
   - small (470MB) - Tốt hơn với tiếng Việt
   - medium (1.5GB) - Accuracy cao nhất

YÊU CẦU PHẦN CỨNG TỐI THIỂU:
- CPU: 4+ cores
- RAM: 16GB (32GB recommended)
- GPU: 6GB+ VRAM (RTX 3060 trở lên)
- Disk: 20GB free space

YÊU CẦU PHẦN CỨNG KHUYẾN NGHỊ:
- CPU: 8+ cores
- RAM: 32GB+
- GPU: 12GB+ VRAM (RTX 4070 trở lên)
- Disk: 50GB+ free space
"""
