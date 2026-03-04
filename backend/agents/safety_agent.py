"""
Safety Monitor Agent - Giám sát an toàn thuốc và cảnh báo
"""
from typing import List
import re
from backend.models import AgentType, ChatRequest, ChatResponse
from backend.agents.base_agent import BaseAgent
from backend.utils import app_logger


class SafetyAgent(BaseAgent):
    """Agent chuyên giám sát an toàn thuốc"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.SAFETY_MONITOR,
            name="Safety Monitor Agent",
            description="Giám sát an toàn thuốc và cảnh báo"
        )
        
        self.safety_keywords = [
            'tác dụng phụ', 'side effect', 'phản ứng', 'an toàn',
            'nguy hiểm', 'độc', 'toxic', 'cảnh báo', 'warning',
            'chống chỉ định', 'contraindication', 'không nên',
            'dị ứng', 'allergy', 'thai phụ', 'pregnant', 'cho con bú'
        ]
    
    async def can_handle(self, request: ChatRequest) -> bool:
        """Kiểm tra yêu cầu có phải về an toàn thuốc không"""
        message_lower = request.message.lower()
        
        # Kiểm tra từ khóa an toàn
        has_keyword = any(
            keyword in message_lower 
            for keyword in self.safety_keywords
        )
        
        # Kiểm tra các mẫu liên quan an toàn
        safety_patterns = [
            r'(có|bị)\s+(tác dụng phụ|phản ứng)',
            r'(an toàn|nguy hiểm)\s+(không|hay)',
            r'(dùng|sử dụng)\s+(được|có được)\s+(không)',
        ]
        
        has_pattern = any(
            re.search(pattern, message_lower)
            for pattern in safety_patterns
        )
        
        return has_keyword or has_pattern
    
    async def process(self, request: ChatRequest) -> ChatResponse:
        """Xử lý yêu cầu giám sát an toàn"""
        app_logger.info(f"Đang xử lý kiểm tra an toàn: {request.message[:100]}")
        
        try:
            # Trích xuất tên thuốc và mối quan ngại an toàn
            drug_name = self._extract_drug_name(request.message)
            safety_concern = self._identify_safety_concern(request.message)
            
            # Lấy thông tin an toàn
            if drug_name:
                # Lấy tác dụng phụ
                side_effects_results = await self.retriever.retrieve_side_effects(
                    drug_name=drug_name,
                    top_k=5
                )
                
                # Lấy chống chỉ định
                contra_results = await self.retrieve_context(
                    query=f"chống chỉ định {drug_name}",
                    filter_dict={"type": "contraindication"},
                    top_k=3
                )
                
                results = side_effects_results + contra_results
            else:
                results = await self.retrieve_context(
                    query=request.message,
                    filter_dict={"type": "safety"},
                    top_k=5
                )
            
            # Định dạng ngữ cảnh
            context = self.format_context(results)
            
            # Tạo phản hồi tập trung vào an toàn
            if context:
                # Cải thiện truy vấn với tập trung an toàn
                enhanced_query = (
                    f"{request.message}\n\n"
                    f"Vui lòng tập trung vào các vấn đề an toàn, tác dụng phụ, "
                    f"và chống chỉ định. Đưa ra cảnh báo rõ ràng nếu cần."
                )
                
                llm_response = await self.generate_response(
                    query=enhanced_query,
                    context=context,
                    conversation_history=self._get_conversation_history(request)
                )
                
                response_message = llm_response['response']
                sources = self.extract_sources(results)
                
                # Tạo cảnh báo
                warnings = self._extract_safety_warnings(
                    results,
                    drug_name,
                    safety_concern
                )
            else:
                response_message = (
                    f"Tôi không tìm thấy thông tin chi tiết về vấn đề an toàn "
                    f"{'của thuốc ' + drug_name if drug_name else 'bạn hỏi'}. "
                    "Tuy nhiên, để đảm bảo an toàn, TÔI KHUYẾN NGHỊ BẠN:\n\n"
                    " Tham khảo bác sĩ hoặc dược sĩ trước khi sử dụng\n"
                    " Đọc kỹ hướng dẫn sử dụng trên bao bì thuốc\n"
                    " Thông báo cho bác sĩ về tiền sử bệnh và các thuốc đang dùng\n"
                    " Ngưng sử dụng và gặp bác sĩ nếu có phản ứng bất thường"
                )
                sources = []
                warnings = self._get_default_safety_warnings()
            
            response = ChatResponse(
                message=response_message,
                conversation_id=request.conversation_id or request.session_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(results),
                sources=sources,
                suggestions=self._get_safety_suggestions(drug_name, safety_concern),
                warnings=warnings,
                metadata={
                    'drug_name': drug_name,
                    'safety_concern': safety_concern,
                    'num_sources': len(results)
                }
            )
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi trong safety agent: {e}")
            raise
    
    def _extract_drug_name(self, message: str) -> str:
        """Trích xuất tên thuốc từ tin nhắn"""
        common_drugs = [
            'paracetamol', 'aspirin', 'ibuprofen', 'amoxicillin',
            'warfarin', 'metformin', 'insulin'
        ]
        
        message_lower = message.lower()
        for drug in common_drugs:
            if drug in message_lower:
                return drug.capitalize()
        
        match = re.search(r'thuốc\s+([a-zA-Z]+)', message_lower)
        if match:
            return match.group(1).capitalize()
        
        return None
    
    def _identify_safety_concern(self, message: str) -> str:
        """Xác định loại mối quan ngại an toàn"""
        message_lower = message.lower()
        
        concerns = {
            'side_effects': ['tác dụng phụ', 'phản ứng', 'side effect'],
            'contraindication': ['chống chỉ định', 'contraindication', 'không nên'],
            'allergy': ['dị ứng', 'allergy'],
            'pregnancy': ['thai phụ', 'có thai', 'pregnant', 'mang thai'],
            'breastfeeding': ['cho con bú', 'breastfeeding', 'đang nuôi con'],
            'overdose': ['quá liều', 'uống nhiều', 'overdose'],
            'interaction': ['tương tác', 'interaction']
        }
        
        for concern_type, keywords in concerns.items():
            if any(keyword in message_lower for keyword in keywords):
                return concern_type
        
        return 'general_safety'
    
    def _extract_safety_warnings(
        self,
        results: List[dict],
        drug_name: str,
        safety_concern: str
    ) -> List[str]:
        """Trích xuất cảnh báo an toàn từ kết quả"""
        warnings = []
        
        # Phân tích kết quả theo mức độ nghiêm trọng
        for result in results:
            doc = result['document'].lower()
            
            # Cảnh báo nghiêm trọng
            if any(word in doc for word in [
                'nghiêm trọng', 'nguy hiểm', 'tử vong', 'severe', 'fatal', 'cấm'
            ]):
                warnings.append(
                    f" CẢNH BÁO NGHIÊM TRỌNG: Vui lòng đọc kỹ các thông tin về "
                    f"{'thuốc ' + drug_name if drug_name else 'thuốc này'}"
                )
            
            # Cảnh báo cho thai phụ
            if 'thai phụ' in doc or 'pregnant' in doc:
                warnings.append(
                    " CẢNH BÁO: Phụ nữ có thai hoặc có thể có thai cần tham khảo bác sĩ"
                )
            
            # Cảnh báo cho trẻ em
            if 'trẻ em' in doc or 'children' in doc:
                warnings.append(
                    " Cần chú ý đặc biệt khi sử dụng cho trẻ em"
                )
        
        # Thêm cảnh báo cụ thể theo mối quan ngại
        if safety_concern == 'allergy':
            warnings.append(
                " Nếu bạn có tiền sử dị ứng với thuốc này hoặc các thuốc tương tự, "
                "KHÔNG sử dụng và thông báo ngay cho bác sĩ"
            )
        elif safety_concern == 'overdose':
            warnings.append(
                " NGUY HIỂM: Quá liều có thể gây hậu quả nghiêm trọng. "
                "Gọi ngay cấp cứu 115 nếu nghi ngờ quá liều"
            )
        
        # Luôn thêm lời nhắc nhở an toàn chung
        warnings.append(
            " Khi có bất kỳ phản ứng bất thường nào, hãy ngưng sử dụng và gặp bác sĩ ngay"
        )
        
        return list(set(warnings))  # Loại bỏ trùng lặp
    
    def _get_default_safety_warnings(self) -> List[str]:
        """Lấy cảnh báo an toàn mặc định"""
        return [
            " Luôn đọc kỹ hướng dẫn sử dụng trước khi dùng thuốc",
            " Báo cho bác sĩ/dược sĩ về tiền sử dị ứng và các thuốc đang dùng",
            " Không tự ý thay đổi liều lượng hoặc ngừng thuốc",
            " Ngưng sử dụng và gặp bác sĩ nếu có phản ứng bất thường"
        ]
    
    def _get_conversation_history(self, request: ChatRequest) -> List[dict]:
        """Lấy lịch sử hội thoại"""
        if request.context and 'history' in request.context:
            return request.context['history']
        return []
    
    def _calculate_confidence(self, results: List[dict]) -> float:
        """Tính điểm tự tin"""
        if not results:
            return 0.4  # Tự tin trung bình ngay cả khi không có kết quả vì an toàn
        
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        return round(avg_similarity, 2)
    
    def _get_safety_suggestions(self, drug_name: str, safety_concern: str) -> List[str]:
        """Lấy gợi ý tiếp theo"""
        suggestions = []
        
        if drug_name:
            suggestions.extend([
                f"Liều lượng an toàn của {drug_name}?",
                f"{drug_name} có tương tác với thuốc nào không?",
                f"Lưu ý gì khi sử dụng {drug_name}?"
            ])
        
        if safety_concern == 'pregnancy':
            suggestions.append("Các thuốc an toàn cho thai phụ?")
        elif safety_concern == 'allergy':
            suggestions.append("Dấu hiệu dị ứng thuốc là gì?")
        
        suggestions.append("Khi nào cần gặp bác sĩ ngay?")
        
        return suggestions
