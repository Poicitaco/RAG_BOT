"""
Interaction Check Agent - Kiểm tra tương tác thuốc
"""
from typing import List
import re
from backend.models import AgentType, ChatRequest, ChatResponse, InteractionSeverity
from backend.agents.base_agent import BaseAgent
from backend.utils import app_logger


class InteractionAgent(BaseAgent):
    """Agent chuyên kiểm tra tương tác giữa các loại thuốc"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.INTERACTION_CHECK,
            name="Interaction Check Agent",
            description="Kiểm tra tương tác giữa các loại thuốc"
        )
        
        self.interaction_keywords = [
            'tương tác', 'kết hợp', 'dùng chung', 'uống cùng',
            'conflict', 'interaction', 'với nhau', 'cùng lúc'
        ]
    
    async def can_handle(self, request: ChatRequest) -> bool:
        """Kiểm tra yêu cầu có phải về tương tác thuốc không"""
        message_lower = request.message.lower()
        
        # Kiểm tra từ khóa tương tác
        has_keyword = any(
            keyword in message_lower 
            for keyword in self.interaction_keywords
        )
        
        # Kiểm tra nhiều thuốc được nhắc đến
        drug_count = message_lower.count('thuốc') + message_lower.count('dược')
        has_multiple_drugs = drug_count >= 2 or 'và' in message_lower
        
        return has_keyword or has_multiple_drugs
    
    async def process(self, request: ChatRequest) -> ChatResponse:
        """Xử lý yêu cầu kiểm tra tương tác"""
        app_logger.info(f"Đang xử lý kiểm tra tương tác: {request.message[:100]}")
        
        try:
            # Trích xuất tên thuốc
            drug_names = self._extract_drug_names(request.message)
            
            if len(drug_names) < 2:
                response_message = (
                    "Để kiểm tra tương tác thuốc, vui lòng cho tôi biết tên của ít nhất 2 loại thuốc "
                    "mà bạn muốn sử dụng cùng nhau. Ví dụ: 'Tôi có thể dùng Paracetamol và Ibuprofen cùng lúc không?'"
                )
                return ChatResponse(
                    message=response_message,
                    conversation_id=request.conversation_id or request.session_id,
                    agent_type=self.agent_type,
                    confidence=0.5,
                    suggestions=self._get_interaction_suggestions(drug_names)
                )
            
            # Lấy thông tin tương tác
            results = await self.retriever.retrieve_interactions(
                drug_names=drug_names,
                top_k=7
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
                
                # Trích xuất cảnh báo từ kết quả
                warnings = self._extract_warnings(results, drug_names)
            else:
                # Không tìm thấy tương tác cụ thể
                response_message = (
                    f"Tôi không tìm thấy thông tin cụ thể về tương tác giữa {', '.join(drug_names)}. "
                    "Tuy nhiên, điều này KHÔNG có nghĩa là không có tương tác. "
                    "Tôi KHUYẾN NGHỊ BẠN THAM KHẢO Ý KIẾN BÁC SĨ hoặc DƯỢC SĨ trước khi sử dụng các thuốc này cùng nhau."
                )
                sources = []
                warnings = [
                    " Luôn tham khảo ý kiến chuyên gia trước khi kết hợp nhiều loại thuốc",
                    " Thông báo cho bác sĩ/dược sĩ về tất cả các thuốc bạn đang dùng"
                ]
            
            response = ChatResponse(
                message=response_message,
                conversation_id=request.conversation_id or request.session_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(results),
                sources=sources,
                suggestions=self._get_interaction_suggestions(drug_names),
                warnings=warnings,
                metadata={
                    'drugs_checked': drug_names,
                    'num_drugs': len(drug_names),
                    'num_sources': len(results)
                }
            )
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi trong interaction agent: {e}")
            raise
    
    def _extract_drug_names(self, message: str) -> List[str]:
        """Trích xuất nhiều tên thuốc từ tin nhắn"""
        # Trích xuất đơn giản - có thể cải thiện với NER
        common_drugs = [
            'paracetamol', 'aspirin', 'ibuprofen', 'amoxicillin',
            'metformin', 'omeprazole', 'warfarin', 'digoxin'
        ]
        
        message_lower = message.lower()
        found_drugs = []
        
        for drug in common_drugs:
            if drug in message_lower:
                found_drugs.append(drug.capitalize())
        
        # Look for patterns like "thuốc X và thuốc Y"
        patterns = [
            r'thuốc\s+([a-zA-Z]+)',
            r'([A-Z][a-z]+)\s+và\s+([A-Z][a-z]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                for group in match.groups():
                    if group and len(group) > 2:
                        drug_name = group.capitalize()
                        if drug_name not in found_drugs:
                            found_drugs.append(drug_name)
        
        return found_drugs[:5]  # Giới hạn 5 thuốc
    
    def _extract_warnings(self, results: List[dict], drug_names: List[str]) -> List[str]:
        """Trích xuất cảnh báo từ kết quả tương tác"""
        warnings = []
        
        for result in results:
            # Tìm chỉ số mức độ nghiêm trọng
            doc = result['document'].lower()
            
            if any(word in doc for word in ['nghiêm trọng', 'severe', 'major', 'nguy hiểm']):
                warnings.append(
                    f" CẢNH BÁO: Tương tác nghiêm trọng có thể xảy ra giữa {' và '.join(drug_names)}"
                )
            elif any(word in doc for word in ['trung bình', 'moderate', 'cẩn thận']):
                warnings.append(
                    f" Chú ý: Tương tác mức độ trung bình - cần theo dõi khi sử dụng"
                )
        
        # Luôn thêm cảnh báo an toàn chung
        if not warnings:
            warnings.append(
                " Luôn tham khảo ý kiến bác sĩ/dược sĩ khi kết hợp nhiều loại thuốc"
            )
        
        return list(set(warnings))  # Loại bỏ trùng lặp
    
    def _get_conversation_history(self, request: ChatRequest) -> List[dict]:
        """Lấy lịch sử hội thoại"""
        if request.context and 'history' in request.context:
            return request.context['history']
        return []
    
    def _calculate_confidence(self, results: List[dict]) -> float:
        """Tính điểm tự tin"""
        if not results:
            return 0.3  # Tự tin thấp khi không có kết quả
        
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        return round(avg_similarity, 2)
    
    def _get_interaction_suggestions(self, drug_names: List[str]) -> List[str]:
        """Lấy gợi ý tiếp theo"""
        if len(drug_names) >= 2:
            return [
                f"Liều lượng an toàn khi dùng {' và '.join(drug_names[:2])}?",
                "Khi nào nên uống các thuốc này?",
                "Có lưu ý gì khi sử dụng các thuốc này không?",
                "Tác dụng phụ khi kết hợp các thuốc này?"
            ]
        else:
            return [
                "Tôi muốn kiểm tra tương tác thuốc khác",
                "Có thể cho biết thêm về an toàn khi dùng thuốc?",
            ]
