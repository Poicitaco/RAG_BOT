"""
Cấu hình cho Hệ thống Trợ lý AI Dược phẩm
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Cài đặt ứng dụng từ biến môi trường"""
    
    # Ứng dụng
    APP_NAME: str = "Pharmaceutical AI Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Máy chủ API
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large", 
        env="OPENAI_EMBEDDING_MODEL"
    )
    
    # Anthropic (Alternative)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = Field(
        default="claude-3-opus-20240229", 
        env="ANTHROPIC_MODEL"
    )
    
    # Cơ sở dữ liệu
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="pharma_ai", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    DATABASE_URL: Optional[str] = None
    
    # Redis
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Vector Store
    VECTOR_STORE_TYPE: str = Field(default="chromadb", env="VECTOR_STORE_TYPE")
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/embeddings/chroma", 
        env="CHROMA_PERSIST_DIR"
    )
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    
    # Cài đặt RAG
    EMBEDDING_DIMENSION: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    TOP_K_RESULTS: int = Field(default=5, env="TOP_K_RESULTS")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Cài đặt Agent
    MAX_ITERATIONS: int = Field(default=10, env="MAX_ITERATIONS")
    AGENT_TIMEOUT: int = Field(default=300, env="AGENT_TIMEOUT")
    ENABLE_MEMORY: bool = Field(default=True, env="ENABLE_MEMORY")
    
    # Cài đặt đa phương tiện
    ENABLE_IMAGE_RECOGNITION: bool = Field(default=True, env="ENABLE_IMAGE_RECOGNITION")
    ENABLE_VOICE_INPUT: bool = Field(default=True, env="ENABLE_VOICE_INPUT")
    ENABLE_VOICE_OUTPUT: bool = Field(default=True, env="ENABLE_VOICE_OUTPUT")
    MAX_IMAGE_SIZE: int = Field(default=10485760, env="MAX_IMAGE_SIZE")  # 10MB
    SUPPORTED_IMAGE_FORMATS: str = Field(
        default="jpg,jpeg,png,webp", 
        env="SUPPORTED_IMAGE_FORMATS"
    )
    SUPPORTED_AUDIO_FORMATS: str = Field(
        default="mp3,wav,ogg,m4a", 
        env="SUPPORTED_AUDIO_FORMATS"
    )
    
    # An toàn & Tuân thủ
    ENABLE_SAFETY_CHECK: bool = Field(default=True, env="ENABLE_SAFETY_CHECK")
    ENABLE_INTERACTION_WARNING: bool = Field(
        default=True, 
        env="ENABLE_INTERACTION_WARNING"
    )
    ENABLE_DOSAGE_VALIDATION: bool = Field(
        default=True, 
        env="ENABLE_DOSAGE_VALIDATION"
    )
    REQUIRE_MEDICAL_DISCLAIMER: bool = Field(
        default=True, 
        env="REQUIRE_MEDICAL_DISCLAIMER"
    )
    
    # Giám sát
    ENABLE_PROMETHEUS: bool = Field(default=True, env="ENABLE_PROMETHEUS")
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # Bảo mật
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, 
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        env="CORS_ORIGINS"
    )
    
    # Lưu trữ file
    UPLOAD_DIR: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    MAX_UPLOAD_SIZE: int = Field(default=52428800, env="MAX_UPLOAD_SIZE")  # 50MB
    
    # Giới hạn tốc độ
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PER_HOUR: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    
    @validator("DATABASE_URL", pre=True, always=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        """Tổng hợp URL cơ sở dữ liệu từ các thành phần"""
        if v:
            return v
        return (
            f"postgresql://{values.get('POSTGRES_USER')}:"
            f"{values.get('POSTGRES_PASSWORD')}@"
            f"{values.get('POSTGRES_HOST')}:"
            f"{values.get('POSTGRES_PORT')}/"
            f"{values.get('POSTGRES_DB')}"
        )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Lấy danh sách CORS origins"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def supported_image_formats_list(self) -> List[str]:
        """Lấy danh sách định dạng hình ảnh hỗ trợ"""
        return [fmt.strip() for fmt in self.SUPPORTED_IMAGE_FORMATS.split(",")]
    
    @property
    def supported_audio_formats_list(self) -> List[str]:
        """Lấy danh sách định dạng âm thanh hỗ trợ"""
        return [fmt.strip() for fmt in self.SUPPORTED_AUDIO_FORMATS.split(",")]
    
    def ensure_directories(self) -> None:
        """Đảm bảo các thư mục cần thiết tồn tại"""
        directories = [
            self.CHROMA_PERSIST_DIR,
            self.UPLOAD_DIR,
            "./data/drugs",
            "./data/knowledge",
            "./logs",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instance cài đặt toàn cục
settings = Settings()
settings.ensure_directories()
