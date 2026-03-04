"""
Các dịch vụ đa phương thức cho trợ lý dược phẩm
"""
from .text_service import TextService,get_text_service
from .image_service import ImageService, get_image_service
from .voice_service import VoiceService, get_voice_service

__all__ = [
    "TextService",
    "get_text_service",
    "ImageService",
    "get_image_service",
    "VoiceService",
    "get_voice_service",
]
