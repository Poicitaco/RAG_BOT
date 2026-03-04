"""
Dosage Advisor Agent - Tư vấn liều lượng thuốc
"""
from typing import List, Dict, Any
import re
from backend.models import AgentType, ChatRequest, ChatResponse
from backend.agents.base_agent import BaseAgent
from backend.utils import app_logger


class DosageAgent(BaseAgent):
    """Agent chuyên tư vấn liều lượng thuốc"""
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DOSAGE_ADVISOR,
            name="Dosage Advisor Agent",
            description="Tư vấn liều lượng thuốc"
        )
        
        self.dosage_keywords = [
            'liều', 'liều lượng', 'dose', 'dosage', 'uống bao nhiêu',
            'dùng bao nhiêu', 'ngày mấy lần', 'bao lâu', 'frequency'
        ]
    
    async def can_handle(self, request: ChatRequest) -> bool:
        """Kiểm tra yêu cầu có phải về liều lượng không"""
        message_lower = request.message.lower()
        
        # Kiểm tra từ khóa liều lượng
        has_keyword = any(
            keyword in message_lower 
            for keyword in self.dosage_keywords
        )
        
        # Kiểm tra câu hỏi liên quan liều lượng
        dosage_patterns = [
            r'(uống|dùng|sử dụng)\s+(bao nhiêu|như thế nào|thế nào)',
            r'(liều|dosage)\s+(.*?)(bao nhiêu|nào)',
            r'(ngày|lần)\s+(mấy|bao nhiêu)',
        ]
        
        has_pattern = any(
            re.search(pattern, message_lower)
            for pattern in dosage_patterns
        )
        
        return has_keyword or has_pattern
    
    async def process(self, request: ChatRequest) -> ChatResponse:
        """Xử lý yêu cầu liều lượng"""
        app_logger.info(f"Đang xử lý yêu cầu liều lượng: {request.message[:100]}")
        
        try:
            # Trích xuất tên thuốc và thông tin bệnh nhân
            drug_name = self._extract_drug_name(request.message)
            patient_info = self._extract_patient_info(request)
            
            if not drug_name:
                response_message = (
                    "Vui lòng cho tôi biết tên thuốc mà bạn muốn tư vấn về liều lượng. "
                    "Ví dụ: 'Liều lượng Paracetamol 500mg cho người lớn là bao nhiêu?'"
                )
                return ChatResponse(
                    message=response_message,
                    conversation_id=request.conversation_id or request.session_id,
                    agent_type=self.agent_type,
                    confidence=0.3,
                    suggestions=["Tôi muốn hỏi về liều lượng thuốc cụ thể"]
                )
            
            # Lấy thông tin liều lượng
            results = await self.retriever.retrieve_dosage_info(
                drug_name=drug_name,
                patient_info=patient_info,
                top_k=5
            )
            
            # Định dạng ngữ cảnh
            context = self.format_context(results)
            
            # Cải thiện truy vấn với thông tin bệnh nhân
            enhanced_query = self._enhance_query_with_patient_info(
                request.message,
                patient_info
            )
            
            # Tạo phản hồi
            if context:
                llm_response = await self.generate_response(
                    query=enhanced_query,
                    context=context,
                    conversation_history=self._get_conversation_history(request)
                )
                
                response_message = llm_response['response']
                sources = self.extract_sources(results)
                warnings = self._generate_dosage_warnings(drug_name, patient_info)
            else:
                response_message = (
                    f"Xin lỗi, tôi không tìm thấy thông tin chi tiết về liều lượng {drug_name}. "
                    "Vui lòng tham khảo bác sĩ hoặc dược sĩ để được tư vấn chính xác về liều lượng phù hợp."
                )
                sources = []
                warnings = [
                    " Không tự ý thay đổi liều lượng mà bác sĩ đã kê đơn",
                    " Luôn đọc kỹ hướng dẫn sử dụng trên bao bì thuốc"
                ]
            
            response = ChatResponse(
                message=response_message,
                conversation_id=request.conversation_id or request.session_id,
                agent_type=self.agent_type,
                confidence=self._calculate_confidence(results),
                sources=sources,
                suggestions=self._get_dosage_suggestions(drug_name),
                warnings=warnings,
                metadata={
                    'drug_name': drug_name,
                    'patient_info': patient_info,
                    'num_sources': len(results)
                }
            )
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi trong dosage agent: {e}")
            raise
    
    def _extract_drug_name(self, message: str) -> str:
        """Trích xuất tên thuốc từ tin nhắn"""
        common_drugs = [
            'paracetamol', 'aspirin', 'ibuprofen', 'amoxicillin',
            'metformin', 'vitamin', 'antibiotic'
        ]
        
        message_lower = message.lower()
        for drug in common_drugs:
            if drug in message_lower:
                return drug.capitalize()
        
        # Tìm từ viết hoa có thể là tên thuốc
        match = re.search(r'\b([A-Z][a-z]+(?:cillin|mycin|prazole|statin)?)\b', message)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_patient_info(self, request: ChatRequest) -> Dict[str, Any]:
        """Trích xuất thông tin bệnh nhân từ tin nhắn và ngữ cảnh"""
        patient_info = {}
        message_lower = request.message.lower()
        
        # Trích xuất tuổi
        age_patterns = [
            r'(\d+)\s*(tuổi|năm|years?old)',
            r'(trẻ em|người lớn|người cao tuổi|thai phụ)',
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if match.group(1).isdigit():
                    patient_info['age'] = int(match.group(1))
                else:
                    patient_info['age_group'] = match.group(1)
        
        # Trích xuất cân nặng
        weight_match = re.search(r'(\d+)\s*(kg|kilogram)', message_lower)
        if weight_match:
            patient_info['weight'] = float(weight_match.group(1))
        
        # Kiểm tra ngữ cảnh cho hồ sơ người dùng
        if request.context and 'user_profile' in request.context:
            profile = request.context['user_profile']
            if 'age' in profile and 'age' not in patient_info:
                patient_info['age'] = profile['age']
            if 'weight' in profile and 'weight' not in patient_info:
                patient_info['weight'] = profile['weight']
        
        return patient_info
    
    def _enhance_query_with_patient_info(
        self,
        query: str,
        patient_info: Dict[str, Any]
    ) -> str:
        """Cải thiện truy vấn với thông tin bệnh nhân"""
        enhancements = []
        
        if 'age' in patient_info:
            enhancements.append(f"bệnh nhân {patient_info['age']} tuổi")
        elif 'age_group' in patient_info:
            enhancements.append(f"cho {patient_info['age_group']}")
        
        if 'weight' in patient_info:
            enhancements.append(f"cân nặng {patient_info['weight']}kg")
        
        if enhancements:
            return f"{query}. Thông tin bệnh nhân: {', '.join(enhancements)}"
        
        return query
    
    def _generate_dosage_warnings(
        self,
        drug_name: str,
        patient_info: Dict[str, Any]
    ) -> List[str]:
        """Tạo cảnh báo cụ thể về liều lượng"""
        warnings = []
        
        # Cảnh báo dựa trên tuổi
        if patient_info.get('age'):
            age = patient_info['age']
            if age < 18:
                warnings.append(
                    " Liều lượng cho trẻ em cần được tính toán cẩn thận - "
                    "Tham khảo bác sĩ nhi khoa"
                )
            elif age > 65:
                warnings.append(
                    " Người cao tuổi có thể cần điều chỉnh liều - "
                    "Tham khảo bác sĩ"
                )
        
        # Cảnh báo chung
        warnings.extend([
            " Không tự ý tăng liều lượng mà không có chỉ định của bác sĩ",
            " Uống thuốc đúng giờ và đúng liều để đạt hiệu quả tốt nhất"
        ])
        
        return warnings
    
    def _get_conversation_history(self, request: ChatRequest) -> List[dict]:
        """Lấy lịch sử hội thoại"""
        if request.context and 'history' in request.context:
            return request.context['history']
        return []
    
    def _calculate_confidence(self, results: List[dict]) -> float:
        """Tính điểm tự tin"""
        if not results:
            return 0.2
        
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        return round(avg_similarity, 2)
    
    def _get_dosage_suggestions(self, drug_name: str) -> List[str]:
        """Lấy gợi ý tiếp theo"""
        if drug_name:
            return [
                f"Khi nào nên uống {drug_name}?",
                f"Có thể uống {drug_name} khi đói không?",
                f"Quên uống liều {drug_name} thì xử lý thế nào?",
                f"Tác dụng phụ của {drug_name} là gì?"
            ]
        else:
            return [
                "Tôi muốn hỏi về tương tác thuốc",
                "Có lưu ý gì khi sử dụng thuốc không?",
            ]
