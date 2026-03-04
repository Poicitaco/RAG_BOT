"""
Updated Response Generator hỗ trợ cả OpenAI và Local AI
"""
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from backend.config import settings
from backend.config.local_settings import local_ai_settings
from backend.utils import app_logger, format_medical_disclaimer
from backend.models import AgentType
import time


class ResponseGenerator:
    """Tạo phản hồi sử dụng LLM với context đã truy xuất (hỗ trợ OpenAI + Local)"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        mode: Optional[str] = None
    ):
        """
        Khởi tạo response generator
        
        Args:
            model_name: Tên mô hình LLM
            temperature: Nhiệt độ lấy mẫu
            max_tokens: Số tokens tối đa trong phản hồi
            mode: "local" hoặc "openai" (mặc định từ settings)
        """
        self.mode = mode or local_ai_settings.AI_MODE
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.mode == "local":
            # Sử dụng Ollama local
            from backend.services.local_llm_service import ollama_service
            self.llm = ollama_service
            self.model_name = local_ai_settings.OLLAMA_MODEL
            app_logger.info(f" Generator với Local LLM: {self.model_name}")
        else:
            # Sử dụng OpenAI
            from langchain.chat_models import ChatOpenAI
            self.model_name = model_name or settings.OPENAI_MODEL
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=settings.OPENAI_API_KEY
            )
            app_logger.info(f"  Generator với OpenAI: {self.model_name}")
    
    def _get_system_prompt(self, agent_type: AgentType) -> str:
        """Lấy system prompt dựa trên loại agent"""
        base_prompt = """Bạn là trợ lý AI chuyên về dược phẩm và y tế, hỗ trợ người dân Việt Nam.
Nhiệm vụ của bạn là cung cấp thông tin chính xác, hữu ích về thuốc và sức khỏe.

QUAN TRỌNG:
- LUÔN đưa ra thông tin dựa trên ngữ cảnh được cung cấp
- Trả lời bằng TIẾNG VIỆT, rõ ràng, dễ hiểu
- Nếu không chắc chắn, hãy nói rõ và khuyên người dùng tham khảo bác sĩ
- KHÔNG BAO GIỜ tự ý khuyên dùng thuốc mà không có đủ thông tin
"""
        
        agent_specific_prompts = {
            AgentType.DRUG_INFO: """
CHUYÊN MÔN: Thông tin về thuốc
- Cung cấp thông tin về thành phần, công dụng, dạng bào chế
- Giải thích rõ ràng cách thuốc hoạt động
- Đưa ra thông tin về nhà sản xuất và phân loại thuốc
""",
            AgentType.INTERACTION: """
CHUYÊN MÔN: Tương tác thuốc
- Phân tích tương tác giữa các loại thuốc
- Đánh giá mức độ nghiêm trọng của tương tác
- Đưa ra khuyến cáo cụ thể về việc sử dụng kết hợp
- Cảnh báo các tương tác nguy hiểm
""",
            AgentType.DOSAGE: """
CHUYÊN MÔN: Liều lượng và cách dùng
- Tính toán liều lượng phù hợp theo tuổi, cân nặng
- Hướng dẫn cách sử dụng thuốc đúng cách
- Lưu ý về thời gian và tần suất dùng thuốc
- Tư vấn về việc điều chỉnh liều
""",
            AgentType.SAFETY: """
CHUYÊN MÔN: An toàn và tác dụng phụ
- Cảnh báo về tác dụng phụ có thể xảy ra
- Thông tin về chống chỉ định
- Khuyến cáo đặc biệt cho phụ nữ mang thai, cho con bú
- Hướng dẫn xử lý khi có tác dụng phụ
""",
            AgentType.GENERAL: """
CHUYÊN MÔN: Tư vấn tổng hợp
- Tổng hợp thông tin từ nhiều khía cạnh
- Đưa ra lời khuyên toàn diện
- Ưu tiên an toàn của người dùng
"""
        }
        
        specific_prompt = agent_specific_prompts.get(agent_type, agent_specific_prompts[AgentType.GENERAL])
        return base_prompt + specific_prompt
    
    async def generate(
        self,
        query: str,
        context: str,
        agent_type: AgentType = AgentType.GENERAL,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Tạo phản hồi với context
        
        Args:
            query: Truy vấn của người dùng
            context: Context đã truy xuất từ RAG
            agent_type: Loại agent cho prompts chuyên biệt
            conversation_history: Hội thoại trước đó
            
        Returns:
            Phản hồi đã tạo
        """
        try:
            start_time = time.time()
            
            # Xây dựng system prompt
            system_prompt = self._get_system_prompt(agent_type)
            
            if self.mode == "local":
                # Chế độ Ollama local
                response = await self.llm.generate_with_context(
                    user_message=query,
                    context=context,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history
                )
            else:
                # Chế độ OpenAI
                # Xây dựng messages
                messages = [SystemMessage(content=system_prompt)]
                
                # Thêm lịch sử hội thoại
                if conversation_history:
                    for msg in conversation_history[-5:]:  # 5 tin nhắn cuối
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                
                # Thêm query hiện tại với context
                user_message = f"""Dựa vào thông tin sau đây để trả lời câu hỏi:

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI: {query}

Hãy trả lời bằng tiếng Việt, chính xác và dễ hiểu. Nếu thông tin không đủ, hãy nói rõ."""
                
                messages.append(HumanMessage(content=user_message))
                
                # Tạo
                response = self.llm(messages).content
            
            # Thêm disclaimer y tế
            final_response = response + "\n\n" + format_medical_disclaimer()
            
            elapsed = time.time() - start_time
            app_logger.info(f"Tạo phản hồi trong {elapsed:.2f}s (chế độ: {self.mode})")
            
            return final_response
            
        except Exception as e:
            app_logger.error(f"Lỗi khi tạo phản hồi: {e}")
            raise
    
    async def generate_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Tạo phản hồi đơn giản không có context RAG
        
        Args:
            prompt: Prompt của người dùng
            system_prompt: Hướng dẫn hệ thống tùy chọn
            
        Returns:
            Phản hồi đã tạo
        """
        try:
            if self.mode == "local":
                return await self.llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))
                
                return self.llm(messages).content
                
        except Exception as e:
            app_logger.error(f"Lỗi trong việc tạo đơn giản: {e}")
            raise


# Singleton instance
generator = ResponseGenerator()
