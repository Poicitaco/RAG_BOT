"""
Voice processing API routes - Các endpoint API xử lý giọng nói
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.models import VoiceInputRequest, VoiceInputResponse
from backend.services import get_voice_service
from backend.utils import app_logger, format_response
import base64

router = APIRouter()


@router.post("/recognize", response_model=VoiceInputResponse)
async def process_voice_input(request: VoiceInputRequest):
    """Xử lý đầu vào giọng nói (base64)"""
    try:
        voice_service = get_voice_service()
        response = await voice_service.process_voice_input(
            audio_data=request.audio_data,
            audio_format=request.audio_format,
            session_id=request.session_id,
            language=request.language
        )
        return response
    
    except Exception as e:
        app_logger.error(f"Lỗi khi nhận dạng giọng nói: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-upload")
async def process_voice_upload(
    file: UploadFile = File(...),
    session_id: str = "default",
    language: str = "vi"
):
    """Xử lý đầu vào giọng nói từ file audio upload"""
    try:
        # Đọc file
        contents = await file.read()
        
        # Chuyển sang base64
        audio_data = base64.b64encode(contents).decode('utf-8')
        
        # Lấy extension file
        file_ext = file.filename.split('.')[-1].lower()
        
        # Xử lý audio
        voice_service = get_voice_service()
        response = await voice_service.process_voice_input(
            audio_data=audio_data,
            audio_format=file_ext,
            session_id=session_id,
            language=language
        )
        
        return response
    
    except Exception as e:
        app_logger.error(f"Lỗi khi upload giọng nói: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-formats")
async def get_supported_audio_formats():
    """Lấy danh sách các định dạng audio được hỗ trợ"""
    from backend.config import settings
    return format_response(
        success=True,
        data={
            "formats": settings.supported_audio_formats_list,
            "max_size_mb": settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        }
    )
