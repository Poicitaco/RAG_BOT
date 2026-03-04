"""
Lớp agent cơ sở cho tất cả các agent dược phẩm
"""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from backend.models import AgentType, ChatRequest, ChatResponse
from backend.rag import get_retriever, get_generator
from backend.utils import app_logger
import time


class BaseAgent(ABC):
    """Lớp cơ sở trừa tượng cho tất cả các agent"""
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        description: str
    ):
        """
        Khởi tạo agent cơ sở
        
        Args:
            agent_type: Loại agent
            name: Tên agent
            description: Mô tả agent
        """
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.retriever = get_retriever()
        self.generator = get_generator()
        
        app_logger.info(f"Khởi tạo agent: {name} ({agent_type})")
    
    @abstractmethod
    async def can_handle(self, request: ChatRequest) -> bool:
        """
        Xác định agent này có thể xử lý yêu cầu hay không
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            True nếu agent có thể xử lý, False nếu không
        """
        pass
    
    @abstractmethod
    async def process(self, request: ChatRequest) -> ChatResponse:
        """
        Xử lý yêu cầu và tạo phản hồi
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Phản hồi chat
        """
        pass
    
    async def retrieve_context(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy ngữ cảnh liên quan cho truy vấn
        
        Args:
            query: Chuỗi truy vấn
            filter_dict: Bộ lọc metadata
            top_k: Số kết quả
            
        Returns:
            Danh sách tài liệu được lấy
        """
        try:
            results = await self.retriever.retrieve(
                query=query,
                filter_dict=filter_dict,
                top_k=top_k
            )
            return results
        except Exception as e:
            app_logger.error(f"Lỗi khi lấy ngữ cảnh: {e}")
            return []
    
    async def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Tạo phản hồi sử dụng LLM
        
        Args:
            query: Truy vấn của người dùng
            context: Ngữ cảnh được lấy
            conversation_history: Các tin nhắn trước đó
            
        Returns:
            Phản hồi được tạo kèm metadata
        """
        try:
            result = await self.generator.generate(
                query=query,
                context=context,
                agent_type=self.agent_type,
                conversation_history=conversation_history
            )
            return result
        except Exception as e:
            app_logger.error(f"Lỗi khi tạo phản hồi: {e}")
            raise
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Định dạng kết quả lấy được thành chuỗi ngữ cảnh"""
        return self.retriever.format_context(results)
    
    def extract_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Trích xuất thông tin nguồn từ kết quả"""
        sources = []
        for result in results:
            source = result['metadata'].get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        return sources
    
    def get_suggestions(self, request: ChatRequest) -> List[str]:
        """
        Lấy các gợi ý tiếp theo cho người dùng
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Danh sách gợi ý
        """
        # Gợi ý mặc định, có thể ghi đè bởi lớp con
        return [
            "Tôi có thể hỏi gì khác không?",
            "Bạn có thể cho biết thêm về tác dụng phụ không?",
            "Liều lượng sử dụng như thế nào?",
        ]
    
    async def validate_response(self, response: ChatResponse) -> ChatResponse:
        """
        Xác thực và làm sạch phản hồi trước khi trả về
        
        Args:
            response: Phản hồi cần xác thực
            
        Returns:
            Phản hồi đã được xác thực
        """
        """
        Xác thực và cải thiện phản hồi
        
        Args:
            response: Phản hồi được tạo
            
        Returns:
            Phản hồi đã xác thực
        """
        # Thêm kiểm tra an toàn, cảnh báo, v.v.
        # Có thể ghi đè bởi lớp con
        return response
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.agent_type})"
