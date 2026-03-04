"""
Local Voice Service
Sử dụng faster-whisper thay vì OpenAI Whisper API
"""
import os
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from backend.config.local_settings import local_ai_settings


class LocalVoiceService:
    """Service để transcribe audio local với faster-whisper"""
    
    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """
        Khởi tạo dịch vụ voice local
        
        Args:
            model_size: Kích thước model (tiny, base, small, medium, large-v3)
            device: Device ('cuda', 'cpu')
            compute_type: Loại tính toán ('float16', 'int8', 'int8_float16')
        """
        self.model_size = model_size or local_ai_settings.WHISPER_MODEL_SIZE
        self.device = device or local_ai_settings.WHISPER_DEVICE
        self.compute_type = compute_type or local_ai_settings.WHISPER_COMPUTE_TYPE
        self.language = local_ai_settings.WHISPER_LANGUAGE
        
        # Tự động điều chỉnh device nếu CUDA không khả dụng
        if self.device == "cuda" and not self._is_cuda_available():
            logger.warning("CUDA không khả dụng, chuyển sang CPU")
            self.device = "cpu"
            self.compute_type = "int8"  # CPU tốt hơn với int8
        
        logger.info(f"Khởi tạo Whisper model: {self.model_size}")
        logger.info(f"Device: {self.device}, Compute type: {self.compute_type}")
        
        # Tải model
        self.model = None
        self._load_model()
    
    def _is_cuda_available(self) -> bool:
        """Kiểm tra CUDA có khả dụng không"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=local_ai_settings.MODEL_CACHE_DIR
            )
            logger.info(f" Đã load Whisper model: {self.model_size}")
            
        except Exception as e:
            logger.error(f" Không thể load Whisper model: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Đường dẫn file audio
            language: Ngôn ngữ (vi, en, auto)
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"File không tồn tại: {audio_path}")
            
            if not self.model:
                raise Exception("Model chưa được load")
            
            language = language or self.language
            
            # Chuyển đổi giọng nói
            logger.info(f"Đang transcribe: {audio_path}")
            segments, info = self.model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection để lọc noise
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
            
            # Trích xuất text từ segments
            full_text = ""
            segment_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            
            full_text = full_text.strip()
            
            # Metadata
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments_count": len(segment_list),
                "segments": segment_list
            }
            
            logger.info(f" Transcribed: {len(full_text)} ký tự")
            logger.info(f"   Language: {info.language} ({info.language_probability:.2%})")
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Lỗi khi transcribe audio: {e}")
            raise
    
    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        temp_dir: str = "./temp",
        language: Optional[str] = None
    ) -> Tuple[str, dict]:
        """
        Transcribe audio từ bytes
        
        Args:
            audio_bytes: Audio data dạng bytes
            temp_dir: Thư mục tạm để lưu file
            language: Ngôn ngữ
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        try:
            # Tạo thư mục tạm
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            
            # Lưu vào file tạm
            import uuid
            temp_file = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}.wav")
            
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            
            # Chuyển đổi giọng nói
            text, metadata = await self.transcribe_audio(temp_file, language)
            
            # Dọn dẹp
            try:
                os.remove(temp_file)
            except:
                pass
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Lỗi khi transcribe audio bytes: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Lấy thông tin về model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language
        }


# Singleton instance
local_voice_service = LocalVoiceService()


# ============= HELPER FUNCTIONS =============

async def test_voice_service(audio_file: str):
    """Test voice service với audio file"""
    print(f" Testing Local Voice Service...")
    print(f"Audio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f" File không tồn tại!")
        return
    
    # Model info
    info = local_voice_service.get_model_info()
    print(f"\n Model info:")
    print(f"   Size: {info['model_size']}")
    print(f"   Device: {info['device']}")
    print(f"   Compute type: {info['compute_type']}")
    
    # Transcribe
    print(f"\n Transcribing...")
    text, metadata = await local_voice_service.transcribe_audio(audio_file)
    
    print(f"\n Transcription result:")
    print(f"   Text: {text}")
    print(f"   Language: {metadata['language']} ({metadata['language_probability']:.2%})")
    print(f"   Duration: {metadata['duration']:.2f}s")
    print(f"   Segments: {metadata['segments_count']}")


if __name__ == "__main__":
    import sys
    import asyncio
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        asyncio.run(test_voice_service(audio_file))
    else:
        print("Usage: python local_voice_service.py <audio_file>")
