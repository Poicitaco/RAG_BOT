"""
Local LLM Service sử dụng Ollama
Thay thế OpenAI API bằng models chạy local
"""
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger

from backend.config.local_settings import local_ai_settings


class OllamaService:
    """Service để tương tác với Ollama local LLM"""
    
    def __init__(self):
        self.base_url = local_ai_settings.OLLAMA_BASE_URL
        self.model = local_ai_settings.OLLAMA_MODEL
        self.timeout = local_ai_settings.OLLAMA_TIMEOUT
        self.temperature = local_ai_settings.OLLAMA_TEMPERATURE
        self.max_tokens = local_ai_settings.OLLAMA_MAX_TOKENS
        
    async def is_available(self) -> bool:
        """Kiểm tra xem Ollama server có đang chạy không"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server không khả dụng: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """Liệt kê các models đã tải về"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Tải model về (nếu chưa có)"""
        try:
            logger.info(f"Đang tải model {model_name}...")
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                )
                response.raise_for_status()
                logger.info(f"Đã tải xong model {model_name}")
                return True
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text từ prompt
        
        Args:
            prompt: Prompt từ người dùng
            system_prompt: Hướng dẫn hệ thống
            temperature: Ghi đè temperature mặc định
            max_tokens: Ghi đè max tokens mặc định
            stream: Streaming response (chưa triển khai)
        """
        try:
            # Chuẩn bị messages
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Chuẩn bị request
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                }
            }
            
            # Gọi Ollama API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                return result["message"]["content"].strip()
                
        except httpx.TimeoutException:
            logger.error("Ollama request timeout")
            raise Exception("LLM request timeout. Model có thể đang quá tải.")
        except Exception as e:
            logger.error(f"Lỗi khi generate với Ollama: {e}")
            raise
    
    async def generate_with_context(
        self,
        user_message: str,
        context: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate với context từ RAG và conversation history
        
        Args:
            user_message: Câu hỏi của user
            context: Context từ RAG retrieval
            system_prompt: System instruction
            conversation_history: Lịch sử hội thoại (list of {role, content})
        """
        # Xây dựng full prompt với context
        full_prompt = f"""Dựa vào thông tin sau đây để trả lời câu hỏi:

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {user_message}

Hãy trả lời bằng tiếng Việt, chính xác và dễ hiểu. Nếu thông tin không đủ, hãy nói rõ."""

        # Thêm conversation history nếu có
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history[-5:])  # Chỉ lấy 5 messages gần nhất
        
        messages.append({"role": "user", "content": full_prompt})
        
        # Gọi API
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                return result["message"]["content"].strip()
                
        except Exception as e:
            logger.error(f"Lỗi khi generate with context: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho text bằng Ollama
        
        Args:
            text: Text cần embed
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            payload = {
                "model": local_ai_settings.OLLAMA_EMBEDDING_MODEL,
                "prompt": text
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                return result["embedding"]
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {e}")
            raise


# Singleton instance
ollama_service = OllamaService()


async def check_ollama_setup():
    """Helper để kiểm tra Ollama setup"""
    print(" Kiểm tra Ollama setup...")
    
    # Check availability
    available = await ollama_service.is_available()
    if not available:
        print(" Ollama server không chạy!")
        print("   Hãy chạy: ollama serve")
        return False
    
    print(" Ollama server đang chạy")
    
    # Check models
    models = await ollama_service.list_models()
    print(f"\n Models đã tải về: {models}")
    
    # Check if required model is available
    if local_ai_settings.OLLAMA_MODEL not in models:
        print(f"\n  Model {local_ai_settings.OLLAMA_MODEL} chưa được tải về")
        print(f"   Đang tải model...")
        success = await ollama_service.pull_model(local_ai_settings.OLLAMA_MODEL)
        if success:
            print(" Đã tải xong model!")
        else:
            print(" Không thể tải model")
            return False
    
    # Check embedding model
    if local_ai_settings.OLLAMA_EMBEDDING_MODEL not in models:
        print(f"\n  Embedding model {local_ai_settings.OLLAMA_EMBEDDING_MODEL} chưa có")
        print(f"   Đang tải...")
        success = await ollama_service.pull_model(local_ai_settings.OLLAMA_EMBEDDING_MODEL)
        if success:
            print(" Đã tải xong embedding model!")
        else:
            print(" Không thể tải embedding model")
            return False
    
    print("\n Ollama đã sẵn sàng!")
    return True


if __name__ == "__main__":
    # Test
    asyncio.run(check_ollama_setup())
