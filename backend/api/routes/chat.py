"""
Chat API routes - Các endpoint API chat
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from backend.models import ChatRequest, ChatResponse
from backend.services import get_text_service
from backend.utils import app_logger, format_response

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat với trợ lý AI
    
    Xử lý tin nhắn người dùng và trả về phản hồi AI
    """
    try:
        text_service = get_text_service()
        response = await text_service.process_message(
            message=request.message,
            session_id=request.session_id,
            conversation_id=request.conversation_id,
            context=request.context
        )
        return response
    
    except Exception as e:
        app_logger.error(f"Lỗi trong chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """Lấy lịch sử chat cho một session"""
    try:
        # TODO: Implement conversation history retrieval
        return format_response(
            success=True,
            data=[],
            message="Lấy lịch sử chat thành công"
        )
    except Exception as e:
        app_logger.error(f"Lỗi khi lấy lịch sử chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Xóa một session chat"""
    try:
        # TODO: Implement session clearing
        return format_response(
            success=True,
            message="Xóa session thành công"
        )
    except Exception as e:
        app_logger.error(f"Lỗi khi xóa session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
