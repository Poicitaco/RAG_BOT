"""
Generator phản hồi RLHF
Sử dụng reward model để cải thiện chất lượng phản hồi thông qua reinforcement learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger

from backend.models.reward_model import RewardModel
from backend.services.ai_adapter import AIAdapter


@dataclass
class GeneratorExperience:
    """Dữ liệu kinh nghiệm cho quá trình huấn luyện RL"""
    query: str
    query_embedding: np.ndarray
    generated_response: str
    response_embedding: np.ndarray
    reward: float
    baseline_reward: float  # Để tính advantage


class RLHFGenerator:
    """
    Generator phản hồi được tăng cường bởi RLHF
    
    Sử dụng reinforcement learning để tối ưu hóa chất lượng phản hồi dựa trên
    dự đoán của reward model và phản hồi từ người dùng.
    
    Thuật toán: Policy Gradient (REINFORCE) với baseline
    - Tạo nhiều phản hồi ứng viên
    - Đánh giá bằng reward model
    - Cập nhật policy để ưu tiên phản hồi có reward cao
    - Liên tục cải thiện từ phản hồi thực tế
    """
    
    def __init__(
        self,
        ai_adapter: AIAdapter,
        reward_model: RewardModel,
        learning_rate: float = 1e-5,
        num_candidates: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Khởi tạo RLHF generator
        
        Args:
            ai_adapter: AI adapter cho LLM và embeddings
            reward_model: Reward model đã được huấn luyện
            learning_rate: Learning rate cho cập nhật policy
            num_candidates: Số lượng phản hồi ứng viên cần tạo
            device: 'cuda' hoặc 'cpu'
        """
        self.ai_adapter = ai_adapter
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        
        self.num_candidates = num_candidates
        self.device = device
        
        # Buffer kinh nghiệm
        self.experience_buffer: List[GeneratorExperience] = []
        self.max_buffer_size = 10000
        
        # Thống kê
        self.generation_count = 0
        self.avg_reward = 0.0
        
        logger.info(f"RLHF Generator initialized (candidates={num_candidates}, device={device})")
    
    async def generate_response(
        self,
        query: str,
        context: List[str],
        conversation_history: List[Dict[str, str]] = None,
        use_rlhf: bool = True
    ) -> Tuple[str, float]:
        """
        Tạo phản hồi tối ưu sử dụng RLHF
        
        Args:
            query: Câu hỏi từ người dùng
            context: Context đã được truy xuất từ RAG
            conversation_history: Lịch sử hội thoại trước đó
            use_rlhf: Có sử dụng tối ưu RLHF hay không
            
        Returns:
            (best_response, reward_score)
        """
        if not use_rlhf:
            # Fallback về generation tiêu chuẩn
            response = await self._generate_single_response(query, context, conversation_history)
            return response, 0.0
        
        # Tạo query embedding
        query_embedding = await self.ai_adapter.embedding_service.embed_query(query)
        
        # Tạo nhiều phản hồi ứng viên
        candidates = await self._generate_candidates(query, context, conversation_history)
        
        # Đánh giá ứng viên bằng reward model
        best_response, best_reward, candidate_scores = await self._select_best_candidate(
            query_embedding,
            candidates
        )
        
        # Lưu kinh nghiệm
        response_embedding = await self.ai_adapter.embedding_service.embed_query(best_response)
        baseline_reward = np.mean(candidate_scores)
        
        experience = GeneratorExperience(
            query=query,
            query_embedding=query_embedding,
            generated_response=best_response,
            response_embedding=response_embedding,
            reward=best_reward,
            baseline_reward=baseline_reward
        )
        
        self._add_experience(experience)
        
        # Cập nhật thống kê
        self.generation_count += 1
        self.avg_reward = (self.avg_reward * 0.99) + (best_reward * 0.01)
        
        logger.info(
            f"Generated response (reward: {best_reward:.3f}, "
            f"avg: {self.avg_reward:.3f}, "
            f"count: {self.generation_count})"
        )
        
        return best_response, best_reward
    
    async def _generate_candidates(
        self,
        query: str,
        context: List[str],
        conversation_history: List[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Tạo nhiều phản hồi ứng viên
        
        Sử dụng temperature sampling để tạo phản hồi đa dạng
        """
        candidates = []
        
        # Tạo với các temperature khác nhau để tăng tính đa dạng
        temperatures = np.linspace(0.5, 1.2, self.num_candidates)
        
        for temp in temperatures:
            response = await self._generate_single_response(
                query,
                context,
                conversation_history,
                temperature=float(temp)
            )
            candidates.append(response)
        
        return candidates
    
    async def _generate_single_response(
        self,
        query: str,
        context: List[str],
        conversation_history: List[Dict[str, str]] = None,
        temperature: float = 0.7
    ) -> str:
        """Tạo một phản hồi duy nhất sử dụng LLM"""
        
        # Xây dựng prompt
        system_prompt = """Bạn là trợ lý AI chuyên về dược phẩm.
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.
Trả lời bằng tiếng Việt, chính xác và hữu ích."""
        
        context_text = "\n\n".join(context) if context else "Không có thông tin."
        
        prompt = f"""Thông tin tham khảo:
{context_text}

Câu hỏi: {query}

Trả lời:"""
        
        # Tạo phản hồi
        response = await self.ai_adapter.llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=500
        )
        
        return response
    
    async def _select_best_candidate(
        self,
        query_embedding: np.ndarray,
        candidates: List[str]
    ) -> Tuple[str, float, List[float]]:
        """
        Chọn ứng viên tốt nhất sử dụng reward model
        
        Returns:
            (best_response, best_reward, all_scores)
        """
        scores = []
        
        # Đánh giá từng ứng viên
        for candidate in candidates:
            # Lấy response embedding
            response_embedding = await self.ai_adapter.embedding_service.embed_query(candidate)
            
            # Dự đoán reward
            reward = self.reward_model.predict_reward(query_embedding, response_embedding)
            scores.append(reward)
        
        # Chọn tốt nhất
        best_idx = np.argmax(scores)
        best_response = candidates[best_idx]
        best_reward = scores[best_idx]
        
        return best_response, best_reward, scores
    
    def _add_experience(self, experience: GeneratorExperience):
        """Thêm kinh nghiệm vào buffer"""
        self.experience_buffer.append(experience)
        
        # Giới hạn kích thước buffer
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def update_from_human_feedback(
        self,
        query: str,
        response: str,
        human_rating: float
    ):
        """
        Cập nhật từ phản hồi trực tiếp của người dùng
        
        Args:
            query: Câu hỏi gốc
            response: Phản hồi đã được tạo
            human_rating: Đánh giá từ người dùng (-1 đến +1)
        """
        # Tìm kinh nghiệm phù hợp
        for exp in reversed(self.experience_buffer):
            if exp.query == query and exp.generated_response == response:
                # Cập nhật reward với phản hồi từ người dùng
                # Kết hợp predicted reward với human rating
                exp.reward = 0.7 * exp.reward + 0.3 * human_rating
                
                logger.info(
                    f"Updated experience with human feedback: "
                    f"rating={human_rating:.2f}, "
                    f"new_reward={exp.reward:.2f}"
                )
                break
    
    def get_statistics(self) -> Dict[str, float]:
        """Lấy thống kê generation"""
        if not self.experience_buffer:
            return {
                "generation_count": 0,
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "buffer_size": 0
            }
        
        rewards = [exp.reward for exp in self.experience_buffer]
        
        return {
            "generation_count": self.generation_count,
            "avg_reward": self.avg_reward,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "buffer_size": len(self.experience_buffer),
            "recent_avg_reward": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }
    
    def save_experiences(self, path: str):
        """Lưu kinh nghiệm vào file"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.experience_buffer, f)
        logger.info(f"Saved {len(self.experience_buffer)} experiences to {path}")
    
    def load_experiences(self, path: str):
        """Nạp kinh nghiệm từ file"""
        import pickle
        with open(path, 'rb') as f:
            self.experience_buffer = pickle.load(f)
        logger.info(f"Loaded {len(self.experience_buffer)} experiences from {path}")


# Demo
async def demo():
    """Demo RLHF generator"""
    from backend.config.settings import get_settings
    
    print("\n" + "="*70)
    print("DEMO RLHF GENERATOR")
    print("="*70)
    
    # Khởi tạo
    settings = get_settings()
    ai_adapter = AIAdapter(settings=settings)
    
    # Tạo reward model
    reward_model = RewardModel(embedding_dim=768)
    
    # Tạo RLHF generator
    generator = RLHFGenerator(
        ai_adapter=ai_adapter,
        reward_model=reward_model,
        num_candidates=3
    )
    
    # Câu hỏi test
    query = "Paracetamol là thuốc gì?"
    context = [
        "Paracetamol là thuốc giảm đau, hạ sốt phổ biến.",
        "Liều dùng: 500-1000mg, 4-6 giờ/lần."
    ]
    
    print(f"\nQuery: {query}")
    print(f"Context: {len(context)} chunks")
    
    # Tạo với RLHF
    response, reward = await generator.generate_response(query, context, use_rlhf=True)
    
    print(f"\nResponse (reward={reward:.3f}):")
    print(f"   {response[:200]}...")
    
    # Mô phỏng phản hồi từ người dùng
    human_rating = 0.8
    generator.update_from_human_feedback(query, response, human_rating)
    
    print(f"\nHuman feedback: {human_rating:+.1f}")
    
    # Thống kê
    stats = generator.get_statistics()
    print(f"\nThống kê:")
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
    
    print("\nDemo hoàn thành!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
