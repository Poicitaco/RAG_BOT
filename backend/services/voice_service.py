"""
Voice processing service for speech-to-text and text-to-speech - Dịch vụ xử lý giọng nói
"""
from typing import Optional
import base64
from io import BytesIO
import openai
from backend.models import VoiceInputRequest, VoiceInputResponse, ChatRequest
from backend.agents import get_orchestrator
from backend.utils import app_logger
from backend.config import settings


class VoiceService:
    """Dịch vụ xử lý đầu vào và đầu ra giọng nói"""
    
    def __init__(self):
        """Khởi tạo voice service"""
        openai.api_key = settings.OPENAI_API_KEY
        self.orchestrator = get_orchestrator()
        app_logger.info("Initialized voice service")
    
    async def process_voice_input(
        self,
        audio_data: str,
        audio_format: str,
        session_id: str,
        language: str = "vi"
    ) -> VoiceInputResponse:
        """
        Xử lý đầu vào giọng nói: speech-to-text sau đó tạo phản hồi
        
        Args:
            audio_data: Audio mã hóa Base64
            audio_format: Định dạng audio (mp3, wav, v.v.)
            session_id: ID phiên
            language: Mã ngôn ngữ
            
        Returns:
            Phản hồi voice input với transcription và chat response
        """
        try:
            # Chuyển đổi audio thành văn bản
            transcribed_text = await self._transcribe_audio(
                audio_data,
                audio_format,
                language
            )
            
            if not transcribed_text:
                return VoiceInputResponse(
                    transcribed_text="",
                    chat_response=None,
                    message="Không thể nhận diện được giọng nói"
                )
            
            # Xử lý văn bản đã chuyển đổi
            request = ChatRequest(
                message=transcribed_text,
                session_id=session_id,
                message_type="voice"
            )
            
            chat_response = await self.orchestrator.process_request(request)
            
            return VoiceInputResponse(
                transcribed_text=transcribed_text,
                chat_response=chat_response
            )
            
        except Exception as e:
            app_logger.error(f"Lỗi khi xử lý đầu vào giọng nói: {e}")
            raise
    
    async def _transcribe_audio(
        self,
        audio_data: str,
        audio_format: str,
        language: str
    ) -> str:
        """
        Chuyển đổi audio thành văn bản sử dụng Whisper API
        
        Args:
            audio_data: Audio mã hóa Base64
            audio_format: Định dạng audio
            language: Mã ngôn ngữ
            
        Returns:
            Văn bản đã chuyển đổi
        """
        try:
            # Giải mã audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Tạo file-like object
            audio_file = BytesIO(audio_bytes)
            audio_file.name = f"audio.{audio_format}"
            
            # Sử dụng OpenAI Whisper API
            response = await openai.Audio.atranscribe(
                model="whisper-1",
                file=audio_file,
                language=language if language != "vi" else "vi"
            )
            
            transcribed_text = response.get('text', '').strip()
            app_logger.info(f"Transcribed audio: {transcribed_text[:100]}")
            
            return transcribed_text
            
        except Exception as e:
            app_logger.error(f"Lỗi khi chuyển đổi audio: {e}")
            return ""
    
    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        language: str = "vi"
    ) -> bytes:
        """
        Chuyển đổi văn bản thành giọng nói sử dụng TTS API
        
        Args:
            text: Văn bản cần chuyển đổi
            voice: Giọng nói sử dụng
            language: Mã ngôn ngữ
            
        Returns:
            Bytes audio
        """
        try:
            # Sử dụng OpenAI TTS API
            response = await openai.Audio.atranscribe(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            return response.content
            
        except Exception as e:
            app_logger.error(f"Lỗi trong text-to-speech: {e}")
            raise


# Global instance
_voice_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """Lấy hoặc tạo voice service toàn cục"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
