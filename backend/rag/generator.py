"""
Response generation for RAG system - Tạo phản hồi cho hệ thống RAG
"""
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.config import settings
from backend.utils import app_logger, format_medical_disclaimer
from backend.models import AgentType
import time


class ResponseGenerator:
    """Tạo phản hồi sử dụng LLM với context đã truy xuất"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Khởi tạo response generator
        
        Args:
            model_name: Tên mô hình LLM
            temperature: Nhiệt độ lấy mẫu
            max_tokens: Số tokens tối đa trong phản hồi
        """
        self.model_name = model_name or settings.OPENAI_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        app_logger.info(f"Initialized response generator with model: {self.model_name}")
    
    def _get_system_prompt(self, agent_type: AgentType) -> str:
        """Lấy system prompt dựa trên loại agent"""
        base_prompt = """Bạn là trợ lý AI chuyên về dược phẩm và y tế, hỗ trợ người dân Việt Nam.
Nhiệm vụ của bạn là cung cấp thông tin chính xác, hữu ích về thuốc và sức khỏe.

QUAN TRỌNG:
- LUÔN đưa ra thông tin dựa trên ngữ cảnh được cung cấp
- Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu
- Luôn nhấn mạnh tầm quan trọng của việc tham khảo ý kiến bác sĩ/dược sĩ
- Không tự ý đưa ra chẩn đoán hoặc khuyến nghị điều trị
- Nếu không chắc chắn, hãy thừa nhận và khuyên người dùng tham khảo chuyên gia
"""
        
        agent_prompts = {
            AgentType.DRUG_INFO: """
CHUYÊN MÔN: Thông tin chi tiết về thuốc
- Cung cấp thông tin về tên thuốc, hoạt chất, công dụng, liều lượng
- Giải thích cách sử dụng thuốc đúng cách
- Nêu rõ các lưu ý quan trọng khi sử dụng
""",
            AgentType.INTERACTION_CHECK: """
CHUYÊN MÔN: Kiểm tra tương tác thuốc
- Phân tích tương tác giữa các loại thuốc
- Đánh giá mức độ nghiêm trọng của tương tác
- Đưa ra khuyến nghị an toàn
- LUÔN cảnh báo về rủi ro tiềm ẩn
""",
            AgentType.DOSAGE_ADVISOR: """
CHUYÊN MÔN: Tư vấn liều lượng thuốc
- Cung cấp thông tin về liều lượng phù hợp
- Xem xét độ tuổi, cân nặng, tình trạng bệnh
- Giải thích cách tính liều lượng
- Nhấn mạnh tầm quan trọng của việc tuân thủ đơn thuốc
""",
            AgentType.SAFETY_MONITOR: """
CHUYÊN MÔN: Giám sát an toàn thuốc
- Cảnh báo về các tác dụng phụ tiềm ẩn
- Nhận diện dấu hiệu nguy hiểm
- Đưa ra khuyến nghị về biện pháp an toàn
- Hướng dẫn khi nào cần gặp bác sĩ ngay
""",
            AgentType.GENERAL: """
CHUYÊN MÔN: Tư vấn chung về dược phẩm
- Trả lời các câu hỏi tổng quát về thuốc và sức khỏe
- Hướng dẫn cách sử dụng thuốc an toàn
- Cung cấp thông tin giáo dục sức khỏe
"""
        }
        
        return base_prompt + agent_prompts.get(agent_type, agent_prompts[AgentType.GENERAL])
    
    async def generate(
        self,
        query: str,
        context: str,
        agent_type: AgentType = AgentType.GENERAL,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        include_disclaimer: bool = True
    ) -> Dict[str, Any]:
        """
        Tạo phản hồi sử dụng LLM
        
        Args:
            query: Truy vấn của người dùng
            context: Context đã truy xuất từ vector store
            agent_type: Loại agent tạo phản hồi
            conversation_history: Các tin nhắn hội thoại trước đó
            include_disclaimer: Có bao gồm disclaimer y tế không
            
        Returns:
            Phản hồi đã tạo với metadata
        """
        start_time = time.time()
        
        try:
            # Xây dựng messages
            messages = []
            
            # System message
            system_prompt = self._get_system_prompt(agent_type)
            messages.append(SystemMessage(content=system_prompt))
            
            # Thêm lịch sử hội thoại
            if conversation_history:
                for msg in conversation_history[-5:]:  # 5 tin nhắn cuối
                    if msg['role'] == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        messages.append(AIMessage(content=msg['content']))
            
            # Thêm context và query
            prompt_with_context = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng:

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {query}

Hãy trả lời một cách chi tiết, chính xác và dễ hiểu. Nếu thông tin không đủ để trả lời, hãy thừa nhận và khuyên người dùng tham khảo bác sĩ/dược sĩ."""
            
            messages.append(HumanMessage(content=prompt_with_context))
            
            # Tạo phản hồi
            response = await self.llm.agenerate([messages])
            generated_text = response.generations[0][0].text
            
            # Thêm disclaimer nếu cần
            if include_disclaimer and settings.REQUIRE_MEDICAL_DISCLAIMER:
                disclaimer = format_medical_disclaimer()
                generated_text = f"{generated_text}\n\n{disclaimer}"
            
            duration = time.time() - start_time
            
            result = {
                'response': generated_text,
                'agent_type': agent_type,
                'model': self.model_name,
                'duration': duration,
                'token_usage': {
                    'prompt_tokens': response.llm_output.get('token_usage', {}).get('prompt_tokens', 0),
                    'completion_tokens': response.llm_output.get('token_usage', {}).get('completion_tokens', 0),
                    'total_tokens': response.llm_output.get('token_usage', {}).get('total_tokens', 0)
                }
            }
            
            app_logger.info(
                f"Generated response in {duration:.3f}s using {result['token_usage']['total_tokens']} tokens"
            )
            
            return result
            
        except Exception as e:
            app_logger.error(f"Lỗi khi tạo phản hồi: {e}")
            raise
    
    async def generate_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Tạo phản hồi đơn giản không có context
        
        Args:
            prompt: Prompt đầu vào
            temperature: Ghi đè temperature tùy chọn
            
        Returns:
            Văn bản đã tạo
        """
        try:
            llm = self.llm
            if temperature is not None:
                llm = ChatOpenAI(
                    model_name=self.model_name,
                    temperature=temperature,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            
            messages = [HumanMessage(content=prompt)]
            response = await llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            app_logger.error(f"Lỗi trong việc tạo đơn giản: {e}")
            raise
    
    async def summarize_conversation(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Tạo tóm tắt cuộc hội thoại
        
        Args:
            conversation_history: Danh sách tin nhắn
            
        Returns:
            Tóm tắt hội thoại
        """
        if not conversation_history:
            return "Cuộc trò chuyện mới"
        
        # Định dạng hội thoại
        formatted_conv = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history
        ])
        
        prompt = f"""Hãy tóm tắt cuộc trò chuyện sau đây thành 1-2 câu ngắn gọn bằng tiếng Việt:

{formatted_conv}

Tóm tắt:"""
        
        return await self.generate_simple(prompt, temperature=0.3)


# Global generator instance
_generator: Optional[ResponseGenerator] = None


def get_generator() -> ResponseGenerator:
    """Lấy hoặc tạo instance generator toàn cục"""
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator
