"""
Local Embedding Service
Sử dụng sentence-transformers thay vì OpenAI embeddings API
"""
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.config.local_settings import local_ai_settings


class LocalEmbeddingService:
    """Service để tạo embeddings local với sentence-transformers"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Khởi tạo dịch vụ embedding local
        
        Args:
            model_name: Tên model (mặc định từ settings)
            device: Device để chạy ('cuda', 'cpu', 'mps')
            cache_dir: Thư mục cache models
        """
        self.model_name = model_name or local_ai_settings.LOCAL_EMBEDDING_MODEL
        self.cache_dir = cache_dir or local_ai_settings.MODEL_CACHE_DIR
        
        # Xác định device
        if device:
            self.device = device
        else:
            if local_ai_settings.USE_GPU:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        
        logger.info(f"Khởi tạo embedding service với model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Tải model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            logger.info(f" Đã load embedding model: {self.model_name}")
            
            # Lấy embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f" Không thể load embedding model: {e}")
            logger.error("Vui lòng kiểm tra tên model hoặc kết nối internet để tải model")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho 1 text
        
        Args:
            text: Text cần embed
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            if not self.model:
                raise Exception("Model chưa được load")
            
            # Mã hóa
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize để tính cosine similarity tốt hơn
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {e}")
            raise
    
    async def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều texts
        
        Args:
            texts: List of texts
            batch_size: Batch size (mặc định từ settings)
            
        Returns:
            List of embedding vectors
        """
        try:
            if not self.model:
                raise Exception("Model chưa được load")
            
            batch_size = batch_size or local_ai_settings.EMBEDDING_BATCH_SIZE
            
            # Mã hóa batch
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=len(texts) > 100,  # Show progress nếu nhiều texts
                normalize_embeddings=True
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings: {e}")
            raise
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Tính cosine similarity giữa 2 texts
        
        Args:
            text1: Text thứ nhất
            text2: Text thứ hai
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Lấy embeddings
            emb1 = await self.embed_text(text1)
            emb2 = await self.embed_text(text2)
            
            # Tính cosine similarity
            emb1_tensor = torch.tensor(emb1)
            emb2_tensor = torch.tensor(emb2)
            
            similarity = torch.nn.functional.cosine_similarity(
                emb1_tensor.unsqueeze(0),
                emb2_tensor.unsqueeze(0)
            )
            
            return float(similarity.item())
            
        except Exception as e:
            logger.error(f"Lỗi khi tính similarity: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Lấy số chiều của embedding vectors"""
        return self.embedding_dim


# Singleton instance
local_embedding_service = LocalEmbeddingService()


# ============= HELPER FUNCTIONS =============

async def test_embedding_service():
    """Test embedding service"""
    print(" Testing Local Embedding Service...")
    
    # Test single text
    text = "Paracetamol là thuốc giảm đau hạ sốt"
    print(f"\nTest text: {text}")
    
    embedding = await local_embedding_service.embed_text(text)
    print(f" Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch
    texts = [
        "Paracetamol giảm đau",
        "Aspirin chống viêm",
        "Hôm nay trời đẹp"
    ]
    print(f"\n Testing batch of {len(texts)} texts...")
    embeddings = await local_embedding_service.embed_texts(texts)
    print(f" Generated {len(embeddings)} embeddings")
    
    # Test similarity
    print(f"\n Testing similarity...")
    sim = await local_embedding_service.compute_similarity(texts[0], texts[1])
    print(f"Similarity (Paracetamol vs Aspirin): {sim:.4f}")
    
    sim2 = await local_embedding_service.compute_similarity(texts[0], texts[2])
    print(f"Similarity (Paracetamol vs Trời đẹp): {sim2:.4f}")
    
    print("\n All tests passed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_embedding_service())
