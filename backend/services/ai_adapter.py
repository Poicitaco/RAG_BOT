"""
AI Adapter - Unified interface cho cả OpenAI API và Local AI
Chọn backend dựa vào AI_MODE trong settings
"""
from typing import List, Optional, Dict, Any
from loguru import logger

from backend.config.local_settings import local_ai_settings


class AIAdapter:
    """
    Adapter pattern để chuyển đổi giữa OpenAI và Local AI
    """
    
    def __init__(self):
        self.mode = local_ai_settings.AI_MODE
        logger.info(f" AI Adapter khởi tạo với mode: {self.mode}")
        
        # Initialize services based on mode
        if self.mode == "local":
            self._init_local_services()
        else:
            self._init_openai_services()
    
    def _init_local_services(self):
        """Khởi tạo các dịch vụ AI local"""
        try:
            from backend.services.local_llm_service import ollama_service
            from backend.services.local_embedding_service import local_embedding_service
            from backend.services.local_voice_service import local_voice_service
            
            self.llm_service = ollama_service
            self.embedding_service = local_embedding_service
            self.voice_service = local_voice_service
            
            logger.info(" Local AI services initialized")
            
        except Exception as e:
            logger.error(f" Không thể init local services: {e}")
            raise
    
    def _init_openai_services(self):
        """Khởi tạo các dịch vụ OpenAI"""
        try:
            # Import các dịch vụ dựa trên OpenAI hiện có
            from backend.rag.embeddings import EmbeddingService
            from backend.services.voice_service import VoiceService
            
            self.embedding_service = EmbeddingService(provider="openai")
            self.voice_service = VoiceService()
            # LLM sẽ dùng từ generator
            
            logger.info(" OpenAI services initialized")
            
        except Exception as e:
            logger.error(f" Không thể init OpenAI services: {e}")
            raise
    
    # ============= LLM Methods =============
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text từ prompt
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Temperature (0-2)
            max_tokens: Max output tokens
            
        Returns:
            Generated text
        """
        if self.mode == "local":
            return await self.llm_service.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # OpenAI mode - sẽ dùng từ ResponseGenerator
            raise NotImplementedError("Use ResponseGenerator for OpenAI mode")
    
    async def generate_with_context(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate với RAG context
        
        Args:
            user_message: User's message
            context: Retrieved context
            system_prompt: System instruction
            conversation_history: Conversation history
            
        Returns:
            Generated response
        """
        if self.mode == "local":
            return await self.llm_service.generate_with_context(
                user_message=user_message,
                context=context,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
        else:
            # Chế độ OpenAI - sẽ dùng từ ResponseGenerator
            raise NotImplementedError("Use ResponseGenerator for OpenAI mode")
    
    # ============= Embedding Methods =============
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return await self.embedding_service.embed_text(text)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        return await self.embedding_service.embed_texts(texts)
    
    # ============= Voice Methods =============
    
    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> tuple:
        """
        Chuyển đổi file audio thành văn bản
        
        Args:
            audio_path: Đường dẫn tới file audio
            language: Mã ngôn ngữ (vi, en, auto)
            
        Returns:
            Tuple của (text, metadata)
        """
        return await self.voice_service.transcribe_audio(audio_path, language)
    
    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> tuple:
        """
        Transcribe audio từ bytes
        
        Args:
            audio_bytes: Audio data
            language: Language code
            
        Returns:
            Tuple of (text, metadata)
        """
        if self.mode == "local":
            return await self.voice_service.transcribe_audio_bytes(audio_bytes, language=language)
        else:
            # Chế độ OpenAI
            return await self.voice_service.transcribe_audio(audio_bytes)
    
    # ============= Utility Methods =============
    
    def get_mode(self) -> str:
        """Lấy chế độ AI hiện tại"""
        return self.mode
    
    def is_local_mode(self) -> bool:
        """Kiểm tra nếu đang chạy ở chế độ local"""
        return self.mode == "local"
    
    def get_model_info(self) -> dict:
        """Lấy thông tin về các mô hình hiện tại"""
        if self.mode == "local":
            return {
                "mode": "local",
                "llm": local_ai_settings.OLLAMA_MODEL,
                "embedding": local_ai_settings.LOCAL_EMBEDDING_MODEL,
                "whisper": local_ai_settings.WHISPER_MODEL_SIZE,
                "device": local_ai_settings.EMBEDDING_DEVICE
            }
        else:
            return {
                "mode": "openai",
                "llm": "gpt-4-turbo-preview",
                "embedding": "text-embedding-3-large",
                "whisper": "whisper-1"
            }


# Singleton instance
ai_adapter = AIAdapter()


# ============= HELPER FUNCTIONS =============

async def check_ai_setup():
    """Kiểm tra AI setup"""
    print(" Kiểm tra AI setup...")
    print(f"Mode: {ai_adapter.get_mode()}")
    
    info = ai_adapter.get_model_info()
    print(f"\n Model info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    if ai_adapter.is_local_mode():
        print("\n Local AI mode - Không cần API key")
        print(" Kiểm tra Ollama...")
        
        # Check Ollama
        from backend.services.local_llm_service import check_ollama_setup
        ollama_ok = await check_ollama_setup()
        
        if not ollama_ok:
            print(" Ollama chưa sẵn sàng!")
            print("\nHướng dẫn:")
            print("1. Cài Ollama: https://ollama.ai/download")
            print("2. Chạy: ollama serve")
            print(f"3. Tải model: ollama pull {local_ai_settings.OLLAMA_MODEL}")
            return False
        
        print("\n Embedding service...")
        try:
            test_text = "Paracetamol là thuốc giảm đau"
            embedding = await ai_adapter.embed_text(test_text)
            print(f"   Dimension: {len(embedding)}")
        except Exception as e:
            print(f" Lỗi: {e}")
            return False
        
        print("\n Whisper service...")
        print(f"   Model: {local_ai_settings.WHISPER_MODEL_SIZE}")
        print(f"   Device: {local_ai_settings.WHISPER_DEVICE}")
        
    else:
        print("\n  OpenAI API mode - Cần API key")
        from backend.config.settings import settings
        if not settings.OPENAI_API_KEY:
            print(" OPENAI_API_KEY chưa được set!")
            return False
        print(" API key có sẵn")
    
    print("\n AI setup hoàn tất!")
    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(check_ai_setup())
