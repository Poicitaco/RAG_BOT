"""
Image processing service for drug recognition - Dịch vụ xử lý ảnh cho nhận dạng thuốc
"""
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from backend.models import DrugRecognitionRequest, DrugRecognitionResponse
from backend.utils import app_logger
from backend.config import settings


class ImageService:
    """Dịch vụ xử lý ảnh và nhận dạng thuốc"""
    
    def __init__(self):
        """Khởi tạo image service"""
        self.max_image_size = settings.MAX_IMAGE_SIZE
        app_logger.info("Initialized image service")
    
    async def recognize_drug(
        self,
        image_data: str,
        image_format: str
    ) -> DrugRecognitionResponse:
        """
        Nhận dạng thuốc từ ảnh sử dụng OCR và vision AI
        
        Args:
            image_data: Ảnh mã hóa Base64
            image_format: Định dạng ảnh (jpg, png, v.v.)
            
        Returns:
            Phản hồi nhận dạng thuốc
        """
        try:
            # Giải mã ảnh
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Xác thực ảnh
            if not self._validate_image(image):
                return DrugRecognitionResponse(
                    success=False,
                    drug_name=None,
                    confidence=0.0,
                    message="Hình ảnh không hợp lệ hoặc quá lớn"
                )
            
            # Trích xuất văn bản sử dụng OCR
            extracted_text = self._extract_text_from_image(image)
            
            if not extracted_text:
                return DrugRecognitionResponse(
                    success=False,
                    drug_name=None,
                    confidence=0.0,
                    extracted_text="",
                    message="Không thể trích xuất văn bản từ hình ảnh"
                )
            
            # Nhận dạng tên thuốc từ văn bản đã trích xuất
            drug_info = self._identify_drug_from_text(extracted_text)
            
            return DrugRecognitionResponse(
                success=True,
                drug_name=drug_info.get('name'),
                confidence=drug_info.get('confidence', 0.5),
                possible_matches=drug_info.get('matches', []),
                extracted_text=extracted_text,
                message="Đã nhận diện được thông tin từ hình ảnh"
            )
            
        except Exception as e:
            app_logger.error(f"Lỗi khi nhận dạng ảnh: {e}")
            return DrugRecognitionResponse(
                success=False,
                drug_name=None,
                confidence=0.0,
                message=f"Lỗi xử lý hình ảnh: {str(e)}"
            )
    
    def _validate_image(self, image: Image.Image) -> bool:
        """Xác thực kích thước và định dạng ảnh"""
        # Check file size (approximate)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        size = buffer.tell()
        
        return size <= self.max_image_size
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Trích xuất văn bản từ ảnh sử dụng OCR"""
        try:
            # Sử dụng Tesseract OCR
            # Lưu ý: Yêu cầu cài đặt tesseract
            text = pytesseract.image_to_string(image, lang='vie+eng')
            return text.strip()
        except Exception as e:
            app_logger.warning(f"Trích xuất OCR thất bại: {e}")
            return ""
    
    def _identify_drug_from_text(self, text: str) -> Dict[str, Any]:
        """Nhận dạng tên thuốc từ văn bản đã trích xuất"""
        # Simple implementation - can be enhanced with NER
        common_drugs = [
            'Paracetamol', 'Aspirin', 'Ibuprofen', 'Amoxicillin',
            'Metformin', 'Omeprazole', 'Amlodipine'
        ]
        
        text_lower = text.lower()
        matches = []
        
        for drug in common_drugs:
            if drug.lower() in text_lower:
                matches.append({
                    'name': drug,
                    'confidence': 0.8
                })
        
        if matches:
            return {
                'name': matches[0]['name'],
                'confidence': matches[0]['confidence'],
                'matches': matches
            }
        
        # Thử trích xuất các từ viết hoa có thể là tên thuốc
        import re
        potential_names = re.findall(r'\b[A-Z][a-z]+(?:cin|mycin|zole|statin)?\b', text)
        
        if potential_names:
            return {
                'name': potential_names[0],
                'confidence': 0.5,
                'matches': [{'name': name, 'confidence': 0.5} for name in potential_names]
            }
        
        return {
            'name': None,
            'confidence': 0.0,
            'matches': []
        }


# Global instance
_image_service: Optional[ImageService] = None


def get_image_service() -> ImageService:
    """Lấy hoặc tạo image service toàn cục"""
    global _image_service
    if _image_service is None:
        _image_service = ImageService()
    return _image_service
