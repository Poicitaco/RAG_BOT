"""
Validation utilities for input data - Tiện ích xác thực dữ liệu đầu vào
"""
import re
from typing import Optional, List
from pathlib import Path


def validate_drug_name(drug_name: str) -> bool:
    """
    Xác thực định dạng tên thuốc
    
    Args:
        drug_name: Tên thuốc cần xác thực
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not drug_name or len(drug_name) < 2:
        return False
    
    # Cho phép chữ, số, khoảng trắng, gạch ngang, và ngoặc
    pattern = r'^[A-Za-z0-9\s\-()]+$'
    return bool(re.match(pattern, drug_name))


def validate_dosage(dosage: str) -> bool:
    """
    Xác thực định dạng liều lượng
    
    Args:
        dosage: Chuỗi liều lượng cần xác thực (ví dụ: "500mg", "2 viên")
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not dosage:
        return False
    
    # Các mẫu liều lượng phổ biến
    patterns = [
        r'^\d+(\.\d+)?\s*(mg|g|ml|mcg|µg|IU|units?)$',  # 500mg, 2.5g, etc.
        r'^\d+\s*(tablet|capsule|pill)s?$',  # 2 tablets
        r'^\d+(\.\d+)?\s*%$',  # 5% (for creams, solutions)
    ]
    
    dosage_lower = dosage.lower().strip()
    return any(re.match(pattern, dosage_lower, re.IGNORECASE) for pattern in patterns)


def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Xác thực phần mở rộng file
    
    Args:
        filename: Tên của file
        allowed_extensions: Danh sách phần mở rộng cho phép (không có dấu chấm)
        
    Returns:
        True nếu hợp lệ, False nếu không
    """
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lstrip('.').lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]


def validate_image_file(filename: str) -> bool:
    """Xác thực loại file ảnh"""
    return validate_file_type(filename, ['jpg', 'jpeg', 'png', 'webp'])


def validate_audio_file(filename: str) -> bool:
    """Xác thực loại file audio"""
    return validate_file_type(filename, ['mp3', 'wav', 'ogg', 'm4a'])


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """
    Làm sạch text input
    
    Args:
        text: Văn bản đầu vào
        max_length: Độ dài tối đa cho phép
        
    Returns:
        Văn bản đã làm sạch
    """
    if not text:
        return ""
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Cắt ngắn nếu quá dài
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_email(email: str) -> bool:
    """Xác thực định dạng email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone_number(phone: str) -> bool:
    """Xác thực số điện thoại (định dạng linh hoạt)"""
    # Remove common separators
    phone = re.sub(r'[\s\-\(\)\+]', '', phone)
    # Check if it's a valid number (8-15 digits)
    return bool(re.match(r'^\d{8,15}$', phone))
