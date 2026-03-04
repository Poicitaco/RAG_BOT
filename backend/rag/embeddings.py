"""
Embedding generation for RAG system - Tạo embedding cho hệ thống RAG
"""
from typing import List, Optional
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from backend.config import settings
from backend.utils import app_logger


class EmbeddingService:
    """Dịch vụ tạo text embedding"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None
    ):
        """
        Khởi tạo dịch vụ embedding
        
        Args:
            provider: Nhà cung cấp embedding ('openai', 'huggingface', 'sentence-transformers')
            model_name: Tên mô hình sử dụng
        """
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.embeddings = self._initialize_embeddings()
        app_logger.info(f"Initialized embedding service: {provider} - {self.model_name}")
    
    def _get_default_model(self) -> str:
        """Lấy tên mô hình mặc định dựa trên provider"""
        defaults = {
            "openai": settings.OPENAI_EMBEDDING_MODEL,
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers": "paraphrase-multilingual-MiniLM-L12-v2",
        }
        return defaults.get(self.provider, defaults["openai"])
    
    def _initialize_embeddings(self) -> Embeddings:
        """Khởi tạo mô hình embedding dựa trên provider"""
        if self.provider == "openai":
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=self.model_name,
            )
        elif self.provider in ["sentence-transformers", "huggingface"]:
            return HuggingFaceEmbeddings(model_name=self.model_name)
        else:
            raise ValueError(f"Nhà cung cấp embedding không được hỗ trợ: {self.provider}")
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một văn bản
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Vector embedding
        """
        try:
            if isinstance(self.embeddings, OpenAIEmbeddings):
                embedding = await self.embeddings.aembed_query(text)
            else:
                embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            app_logger.error(f"Lỗi khi tạo embedding: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embedding cho nhiều văn bản
        
        Args:
            texts: Danh sách văn bản đầu vào
            
        Returns:
            Danh sách vector embedding
        """
        try:
            if isinstance(self.embeddings, OpenAIEmbeddings):
                embeddings = await self.embeddings.aembed_documents(texts)
            else:
                embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            app_logger.error(f"Lỗi khi tạo embeddings: {e}")
            raise
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Tính độ tương đồng cosine giữa hai embedding
        
        Args:
            embedding1: Vector embedding thứ nhất
            embedding2: Vector embedding thứ hai
            
        Returns:
            Điểm tương đồng (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Độ tương đồng cosine
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_embedding_dimension(self) -> int:
        """Lấy số chiều của embeddings"""
        if self.provider == "openai":
            if "3-large" in self.model_name:
                return 3072
            elif "3-small" in self.model_name:
                return 1536
            else:
                return 1536  # Mặc định
        else:
            # Với sentence transformers, cần kiểm tra mô hình
            return 384  # Chiều phổ biến
    

class HuggingFaceEmbeddings(Embeddings):
    """Wrapper cho HuggingFace embeddings"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed một query"""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed nhiều documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Lấy hoặc tạo instance embedding service toàn cục"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(provider="openai")
    return _embedding_service


async def generate_embedding(text: str) -> List[float]:
    """
    Hàm tiện ích để tạo embedding cho văn bản
    
    Args:
        text: Văn bản đầu vào
        
    Returns:
        Vector embedding
    """
    service = get_embedding_service()
    return await service.embed_text(text)


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Hàm tiện ích để tạo embeddings cho nhiều văn bản
    
    Args:
        texts: Danh sách văn bản đầu vào
        
    Returns:
        Danh sách vector embedding
    """
    service = get_embedding_service()
    return await service.embed_texts(texts)
