"""
Image processing API routes - Các endpoint API xử lý ảnh
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.models import DrugRecognitionRequest, DrugRecognitionResponse
from backend.services import get_image_service
from backend.utils import app_logger, format_response
import base64

router = APIRouter()


@router.post("/recognize", response_model=DrugRecognitionResponse)
async def recognize_drug_from_image(request: DrugRecognitionRequest):
    """Nhận dạng thuốc từ ảnh (base64)"""
    try:
        image_service = get_image_service()
        response = await image_service.recognize_drug(
            image_data=request.image_data,
            image_format=request.image_format
        )
        return response
    
    except Exception as e:
        app_logger.error(f"Lỗi khi nhận dạng ảnh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-upload")
async def recognize_drug_from_upload(file: UploadFile = File(...)):
    """Nhận dạng thuốc từ file ảnh upload"""
    try:
        # Đọc file
        contents = await file.read()
        
        # Chuyển sang base64
        image_data = base64.b64encode(contents).decode('utf-8')
        
        # Lấy extension file
        file_ext = file.filename.split('.')[-1].lower()
        
        # Xử lý ảnh
        image_service = get_image_service()
        response = await image_service.recognize_drug(
            image_data=image_data,
            image_format=file_ext
        )
        
        return response
    
    except Exception as e:
        app_logger.error(f"Lỗi khi nhận dạng ảnh upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-formats")
async def get_supported_image_formats():
    """Lấy danh sách các định dạng ảnh được hỗ trợ"""
    from backend.config import settings
    return format_response(
        success=True,
        data={
            "formats": settings.supported_image_formats_list,
            "max_size_mb": settings.MAX_IMAGE_SIZE / (1024 * 1024)
        }
    )
