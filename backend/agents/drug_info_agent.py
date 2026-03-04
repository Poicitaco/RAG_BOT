"""
Drug Information Agent - Cung cấp thông tin chi tiết về thuốc
"""
from typing import List
import re
from backend.models import AgentType, ChatRequest, ChatResponse
from backend.agents.base_agent import BaseAgent
from backend.utils import app_logger


class DrugInfoAgent(BaseAgent):
    """Agent chuyên cung cấp thông tin về thuốc"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DRUG_INFO,
            name="Drug Information Agent",
            description="Cung cấp thông tin chi tiết về thuốc"
        )
        
        # Từ khóa cho biết yêu cầu thông tin thuốc
        self.drug_info_keywords = [
            'thuốc', 'dược', 'paracetamol', 'aspirin', 'ibuprofen',
            'là gì', 'công dụng', 'tác dụng', 'hoạt chất', 'thành phần',
            'sử dụng', 'dùng', 'chỉ định', 'điều trị'
        ]
    
    async def can_handle(self, request: ChatRequest) -> bool:
        """
        Kiểm tra agent này có thể xử lý yêu cầu hay không
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            True nếu yêu cầu về thông tin thuốc
        """
        message_lower = request.message.lower()
        
        # Kiểm tra từ khóa thông tin thuốc
        has_keyword = any(
            keyword in message_lower 
            for keyword in self.drug_info_keywords
        )
        
        # Kiểm tra các mẫu câu hỏi
        question_patterns = [
            r'(.*?)(thuốc|dược)(.*?)(là gì|có tác dụng)',
            r'(.*?)(thông tin|cho biết)(.*?)(thuốc|dược)',
            r'(.*?)(công dụng|tác dụng)(.*?)(thuốc|dược)',
        ]
        
        has_pattern = any(
            re.search(pattern, message_lower)
            for pattern in question_patterns
        )
        
        return has_keyword or has_pattern
    
    async def process(self, request: ChatRequest) -> ChatResponse:
        """
        Xử lý yêu cầu thông tin thuốc
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Phản hồi chat với thông tin thuốc
        """
        app_logger.info(f"Đang xử lý yêu cầu thông tin thuốc: {request.message[:100]}")
        
        try:
            # Trích xuất tên thuốc nếu có thể
            drug_name = self._extract_drug_name(request.message)
            
            # Lấy ngữ cảnh
            if drug_name:
                results = await self.retriever.retrieve_drug_info(
                    drug_name=drug_name,
                    top_k=5
                )
            else:
                results = await self.retrieve_context(
                    query=request.message,
                    filter_dict={"type": "drug_info"},
                    top_k=5
                )
            
            # Định dạng ngữ cảnh
            context = self.format_context(results)
            
            # Tạo phản hồi
            if context:
                llm_response = await self.generate_response(
                    query=request.message,
                    context=context,
                    conversation_history=self._get_conversation_history(request)
                )
                
                response_message = llm_response['response']
                sources = self.extract_sources(results)
            else:
                # Không tìm thấy thông tin liên quan
                response_message = (
                    f"Xin lỗi, tôi không tìm thấy thông tin chi tiết về "
                    f"{'thuốc ' + drug_name if drug_name else 'câu hỏi của bạn'}. "
                    "Bạn có thể cung cấp thêm thông tin hoặc tham khảo bác sĩ/dược sĩ để được tư vấn chính xác hơn."
                )
                sources = []
            
            # Lấy gợi ý
            suggestions = self._get_drug_info_suggestions(drug_name)
            
            # Tạo phản hồi
            response = ChatResponse(
                message=response_message,
                conversation_id=request.conversation_id or request.session_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(results),
                sources=sources,
                suggestions=suggestions,
                metadata={
                    'drug_name': drug_name,
                    'num_sources': len(results)
                }
            )
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi trong drug info agent: {e}")
            raise
    
    def _extract_drug_name(self, message: str) -> str:
        """Trích xuất tên thuốc từ tin nhắn"""
        # Trích xuất đơn giản - tìm từ viết hoa hoặc tên thuốc phổ biến
        # Có thể cải thiện với NER (Named Entity Recognition)
        
        common_drugs = [
            'paracetamol', 'aspirin', 'ibuprofen', 'amoxicillin',
            'metformin', 'omeprazole', 'amlodipine', 'atorvastatin',
            'vitamin', 'antibiotic', 'kháng sinh'
        ]
        
        message_lower = message.lower()
        for drug in common_drugs:
            if drug in message_lower:
                return drug.capitalize()
        
        # Tìm từ sau "thuốc"
        match = re.search(r'thuốc\s+([a-zA-Z0-9]+)', message_lower)
        if match:
            return match.group(1).capitalize()
        
        return None
    
    def _get_conversation_history(self, request: ChatRequest) -> List[dict]:
        """Lấy lịch sử hội thoại từ ngữ cảnh yêu cầu"""
        if request.context and 'history' in request.context:
            return request.context['history']
        return []
    
    def _calculate_confidence(self, results: List[dict]) -> float:
        """Tính điểm tự tin dựa trên kết quả lấy được"""
        if not results:
            return 0.0
        
        # Điểm tương đồng trung bình
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        return round(avg_similarity, 2)
    
    def _get_drug_info_suggestions(self, drug_name: str) -> List[str]:
        """Lấy gợi ý tiếp theo cho thông tin thuốc"""
        suggestions = []
        
        if drug_name:
            suggestions = [
                f"Liều lượng {drug_name} sử dụng như thế nào?",
                f"Tác dụng phụ của {drug_name} là gì?",
                f"{drug_name} có tương tác với thuốc nào không?",
                f"Lưu ý gì khi sử dụng {drug_name}?",
            ]
        else:
            suggestions = [
                "Tôi muốn biết về liều lượng sử dụng",
                "Có tác dụng phụ gì không?",
                "Thuốc này dùng được cho trẻ em không?",
            ]
        
        return suggestions
