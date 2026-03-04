"""
Text processing service - Dịch vụ xử lý văn bản
"""
from typing import Dict, Any, Optional
from backend.models import ChatRequest, ChatResponse
from backend.agents import get_orchestrator
from backend.utils import app_logger, sanitize_input
import time


class TextService:
    """Dịch vụ xử lý hội thoại dựa trên văn bản"""
    
    def __init__(self):
        """Khởi tạo text service"""
        self.orchestrator = get_orchestrator()
        app_logger.info("Initialized text service")
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Xử lý tin nhắn văn bản
        
        Args:
            message: Tin nhắn của người dùng
            session_id: ID phiên
            conversation_id: ID hội thoại tùy chọn
            context: Dữ liệu context tùy chọn
            
        Returns:
            Phản hồi chat
        """
        start_time = time.time()
        
        try:
            # Làm sạch input
            clean_message = sanitize_input(message)
            
            if not clean_message:
                return ChatResponse(
                    message="Vui lòng nhập câu hỏi của bạn.",
                    conversation_id=conversation_id or session_id,
                    agent_type="general",
                    confidence=0.0
                )
            
            # Tạo request
            request = ChatRequest(
                message=clean_message,
                session_id=session_id,
                conversation_id=conversation_id,
                context=context
            )
            
            # Xử lý với orchestrator
            response = await self.orchestrator.process_request(request)
            
            duration = time.time() - start_time
            app_logger.info(f"Processed text message in {duration:.3f}s")
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi khi xử lý tin nhắn văn bản: {e}")
            raise


# Global instance
_text_service: Optional[TextService] = None


def get_text_service() -> TextService:
    """Lấy hoặc tạo text service toàn cục"""
    global _text_service
    if _text_service is None:
        _text_service = TextService()
    return _text_service
