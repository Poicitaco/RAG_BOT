"""
Agent Orchestrator - Điều phối và chuyển yêu cầu đến agent phù hợp
"""
from typing import List, Optional
from backend.models import ChatRequest, ChatResponse, AgentType
from backend.agents.base_agent import BaseAgent
from backend.agents.drug_info_agent import DrugInfoAgent
from backend.agents.interaction_agent import InteractionAgent
from backend.agents.dosage_agent import DosageAgent
from backend.agents.safety_agent import SafetyAgent
from backend.utils import app_logger
import asyncio


class AgentOrchestrator:
    """Điều phối nhiều agent để xử lý yêu cầu người dùng"""
    
    def __init__(self):
        """Khởi tạo orchestrator với tất cả các agent"""
        self.agents: List[BaseAgent] = [
            SafetyAgent(),          # An toàn được u tiên - kiểm tra agent này trước
            InteractionAgent(),     # Kiểm tra tương tác
            DosageAgent(),          # Gợi ý liều lượng
            DrugInfoAgent(),        # Thông tin thuốc tổng quát - khớp rộng nhất
        ]
        
        app_logger.info(f"Đã khởi tạo orchestrator với {len(self.agents)} agent")
    
    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """
        Xử lý yêu cầu người dùng bằng cách chuyển đến agent phù hợp
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Phản hồi chat
        """
        try:
            app_logger.info(f"Đang xử lý yêu cầu: {request.message[:100]}")
            
            # Tìm các agent có khả năng xử lý
            capable_agents = await self._find_capable_agents(request)
            
            if not capable_agents:
                # Không có agent cụ thể - sử dụng fallback tổng quát
                app_logger.info("Không tìm thấy agent cụ thể, sử dụng drug info agent làm fallback")
                agent = self.agents[-1]  # DrugInfoAgent làm fallback
            else:
                # Sử dụng agent đầu tiên (tự tin nhất)
                agent = capable_agents[0]
                app_logger.info(f"Chọn agent: {agent.name}")
            
            # Xử lý với agent đã chọn
            response = await agent.process(request)
            
            # Ghi lại agent được chọn trong metadata
            response.metadata['selected_agent'] = agent.name
            response.metadata['capable_agents'] = [a.name for a in capable_agents]
            
            return response
            
        except Exception as e:
            app_logger.error(f"Lỗi trong orchestrator: {e}")
            # Trả về phản hồi lỗi
            return ChatResponse(
                message=(
                    "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. "
                    "Vui lòng thử lại hoặc liên hệ với chúng tôi để được hỗ trợ."
                ),
                conversation_id=request.conversation_id or request.session_id,
                agent_type=AgentType.GENERAL,
                confidence=0.0,
                warnings=[
                    " Luôn tham khảo ý kiến bác sĩ/dược sĩ cho các vấn đề sức khỏe quan trọng"
                ]
            )
    
    async def _find_capable_agents(self, request: ChatRequest) -> List[BaseAgent]:
        """
        Tìm tất cả các agent có khả năng xử lý yêu cầu
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Danh sách các agent có khả năng, sắp xếp theo ưu tiên
        """
        capable_agents = []
        
        # Kiểm tra khả năng của mỗi agent song song
        tasks = [agent.can_handle(request) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        # Thu thập các agent có khả năng
        for agent, can_handle in zip(self.agents, results):
            if can_handle:
                capable_agents.append(agent)
                app_logger.debug(f"Agent {agent.name} có thể xử lý yêu cầu")
        
        return capable_agents
    
    async def process_multi_intent(
        self,
        request: ChatRequest
    ) -> List[ChatResponse]:
        """
        Xử lý yêu cầu có thể có nhiều ý định
        
        Args:
            request: Yêu cầu chat
            
        Returns:
            Danh sách phản hồi từ các agent khác nhau
        """
        capable_agents = await self._find_capable_agents(request)
        
        if len(capable_agents) <= 1:
            # Ý định đơn - sử dụng xử lý bình thường
            response = await self.process_request(request)
            return [response]
        
        # Nhiều ý định - xử lý với tất cả các agent khả dụng
        app_logger.info(f"Processing multi-intent request with {len(capable_agents)} agents")
        
        tasks = [agent.process(request) for agent in capable_agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Lọc bỏ exceptions
        valid_responses = [
            resp for resp in responses
            if isinstance(resp, ChatResponse)
        ]
        
        return valid_responses
    
    def get_agent_by_type(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        Lấy agent theo loại
        
        Args:
            agent_type: Loại agent
            
        Returns:
            Instance agent hoặc None
        """
        for agent in self.agents:
            if agent.agent_type == agent_type:
                return agent
        return None
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get list of all agents"""
        return self.agents.copy()
    
    def get_agent_info(self) -> List[dict]:
        """Get information about all agents"""
        return [
            {
                'name': agent.name,
                'type': agent.agent_type,
                'description': agent.description
            }
            for agent in self.agents
        ]


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
