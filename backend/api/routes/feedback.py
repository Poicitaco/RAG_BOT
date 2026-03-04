"""
Feedback API Routes
Backend endpoints cho việc thu thập và quản lý feedback người dùng
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from backend.config.settings import Settings, get_settings


router = APIRouter(prefix="/feedback", tags=["feedback"])


# Pydantic models
class FeedbackCreate(BaseModel):
    """Request tạo feedback"""
    query: str = Field(..., description="Truy vấn của người dùng")
    response: str = Field(..., description="Phản hồi của trợ lý")
    rating: float = Field(..., ge=-1, le=1, description="Đánh giá từ -1 đến +1")
    feedback_type: str = Field(default="thumbs", description="Loại: thumbs, star, detailed")
    metadata: Optional[Dict] = Field(default=None, description="Metadata bổ sung")
    text_feedback: Optional[str] = Field(default=None, description="Feedback văn bản tùy chọn")


class FeedbackResponse(BaseModel):
    """Phản hồi feedback"""
    id: str
    query: str
    response: str
    rating: float
    feedback_type: str
    metadata: Optional[Dict]
    text_feedback: Optional[str]
    timestamp: str
    status: str = "received"


class FeedbackStats(BaseModel):
    """Thống kê feedback"""
    total_feedback: int
    positive_count: int
    negative_count: int
    neutral_count: int
    satisfaction_rate: float
    avg_rating: float
    feedback_by_type: Dict[str, int]


class FeedbackStorage:
    """
    Lưu trữ feedback dựa trên file đơn giản
    
    Trong production, nên dùng database thực sự (PostgreSQL, MongoDB, v.v.)
    """
    
    def __init__(self, storage_path: str = "data/feedback"):
        """Khởi tạo storage"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.storage_path / "feedback.jsonl"
        
        logger.info(f"Feedback storage initialized: {self.feedback_file}")
    
    def save_feedback(self, feedback: FeedbackCreate) -> FeedbackResponse:
        """Lưu feedback vào file"""
        # Tạo ID
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Tạo response object
        feedback_response = FeedbackResponse(
            id=feedback_id,
            query=feedback.query,
            response=feedback.response,
            rating=feedback.rating,
            feedback_type=feedback.feedback_type,
            metadata=feedback.metadata,
            text_feedback=feedback.text_feedback,
            timestamp=datetime.now().isoformat()
        )
        
        # Ghi vào file JSONL
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_response.model_dump(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved feedback: {feedback_id} (rating={feedback.rating})")
        
        return feedback_response
    
    def get_all_feedback(self) -> List[FeedbackResponse]:
        """Tải tất cả feedback"""
        if not self.feedback_file.exists():
            return []
        
        feedback_list = []
        
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    feedback_list.append(FeedbackResponse(**data))
        
        return feedback_list
    
    def get_statistics(self) -> FeedbackStats:
        """Lấy thống kê feedback"""
        all_feedback = self.get_all_feedback()
        
        if not all_feedback:
            return FeedbackStats(
                total_feedback=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                satisfaction_rate=0.0,
                avg_rating=0.0,
                feedback_by_type={}
            )
        
        # Đếm theo rating
        positive = sum(1 for fb in all_feedback if fb.rating > 0.3)
        negative = sum(1 for fb in all_feedback if fb.rating < -0.3)
        neutral = sum(1 for fb in all_feedback if -0.3 <= fb.rating <= 0.3)
        
        # Rating trung bình
        avg_rating = sum(fb.rating for fb in all_feedback) / len(all_feedback)
        
        # Tỉ lệ hài lòng (positive / total)
        satisfaction_rate = positive / len(all_feedback) if all_feedback else 0.0
        
        # Đếm theo loại
        feedback_by_type = {}
        for fb in all_feedback:
            feedback_by_type[fb.feedback_type] = feedback_by_type.get(fb.feedback_type, 0) + 1
        
        return FeedbackStats(
            total_feedback=len(all_feedback),
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            satisfaction_rate=satisfaction_rate,
            avg_rating=avg_rating,
            feedback_by_type=feedback_by_type
        )
    
    def export_for_training(self, output_file: str):
        """Xuất feedback theo định dạng phù hợp cho huấn luyện"""
        all_feedback = self.get_all_feedback()
        
        # Chuyển sang định dạng huấn luyện
        training_data = []
        for fb in all_feedback:
            training_data.append({
                "query": fb.query,
                "response": fb.response,
                "rating": fb.rating,
                "metadata": fb.metadata or {}
            })
        
        # Lưu
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(training_data)} feedback items to {output_file}")


# Global storage instance
feedback_storage = FeedbackStorage()


# API Endpoints
@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    settings: Settings = Depends(get_settings)
):
    """
    Gửi feedback của người dùng
    
    Ví dụ:
    ```json
    {
        "query": "Paracetamol là gì?",
        "response": "Paracetamol là thuốc giảm đau...",
        "rating": 0.8,
        "feedback_type": "thumbs",
        "metadata": {"agent": "drug_info", "confidence": 0.95}
    }
    ```
    """
    try:
        feedback_response = feedback_storage.save_feedback(feedback)
        return feedback_response
    
    except Exception as e:
        logger.error(f"Lỗi khi lưu feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=FeedbackStats)
async def get_statistics():
    """
    Lấy thống kê feedback
    
    Trả về thống kê tổng hợp về tất cả feedback
    """
    try:
        stats = feedback_storage.get_statistics()
        return stats
    
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[FeedbackResponse])
async def list_feedback(
    limit: int = 100,
    offset: int = 0,
    rating_filter: Optional[str] = None  # "positive", "negative", "neutral"
):
    """
    Liệt kê tất cả feedback với phân trang
    
    Args:
        limit: Số lượng item tối đa trả về
        offset: Số lượng item bỏ qua
        rating_filter: Lọc theo rating ("positive", "negative", "neutral")
    """
    try:
        all_feedback = feedback_storage.get_all_feedback()
        
        # Lọc theo rating
        if rating_filter:
            if rating_filter == "positive":
                all_feedback = [fb for fb in all_feedback if fb.rating > 0.3]
            elif rating_filter == "negative":
                all_feedback = [fb for fb in all_feedback if fb.rating < -0.3]
            elif rating_filter == "neutral":
                all_feedback = [fb for fb in all_feedback if -0.3 <= fb.rating <= 0.3]
        
        # Áp dụng phân trang
        paginated = all_feedback[offset:offset+limit]
        
        return paginated
    
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_feedback(
    output_file: str = "data/feedback/training_data.json"
):
    """
    Xuất feedback cho huấn luyện mô hình
    
    Lưu feedback theo định dạng phù hợp cho huấn luyện RLHF
    """
    try:
        feedback_storage.export_for_training(output_file)
        
        return {
            "status": "success",
            "message": f"Exported feedback to {output_file}",
            "count": len(feedback_storage.get_all_feedback())
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi xuất feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_feedback():
    """
    Xóa tất cả feedback (sử dụng thận trọng!)
    
    Tạo backup trước khi xóa
    """
    try:
        # Tạo backup
        backup_file = feedback_storage.storage_path / f"feedback_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        if feedback_storage.feedback_file.exists():
            import shutil
            shutil.copy(feedback_storage.feedback_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
            
            # Xóa file gốc
            feedback_storage.feedback_file.unlink()
            logger.info("Cleared feedback file")
        
        return {
            "status": "success",
            "message": "Feedback cleared",
            "backup": str(backup_file)
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi xóa feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@router.get("/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe"""
    return {
        "status": "healthy",
        "storage_path": str(feedback_storage.storage_path),
        "feedback_count": len(feedback_storage.get_all_feedback())
    }
