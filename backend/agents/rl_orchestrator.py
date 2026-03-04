"""
RL Agent Orchestrator - Học cách chọn agent tối ưu
Thay thế rule-based orchestrator bằng RL-based
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import deque
from loguru import logger

from backend.models import AgentType, Message
from backend.agents.base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Mạng neural cho chính sách RL"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass để tạo xác suất hành động"""
        return self.network(state)


class ValueNetwork(nn.Module):
    """Mạng critic để ước tính giá trị state"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Ước tính giá trị của state"""
        return self.network(state)


class Experience:
    """Một kinh nghiệm cho replay buffer"""
    
    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class ReplayBuffer:
    """Bộ đệm kinh nghiệm replay cho huấn luyện"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Thêm kinh nghiệm vào buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Lấy mẫu ngẫu nhiên batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class RLAgentOrchestrator:
    """
    RL-based Agent Orchestrator
    
    Học cách chọn agent tối ưu dựa trên:
    - Query embedding
    - Conversation context
    - Historical performance
    - User feedback
    
    Algorithm: PPO (Proximal Policy Optimization)
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        embedding_service,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Khởi tạo RL Orchestrator
        
        Args:
            agents: Danh sách các agent khả dụng
            embedding_service: Dịch vụ tạo embeddings
            learning_rate: Tốc độ học
            gamma: Hệ số chiết khấu
            epsilon: Tham số PPO clip
            device: Thiết bị chạy
        """
        self.agents = agents
        self.embedding_service = embedding_service
        self.device = device
        
        # Agent mapping
        self.agent_to_idx = {
            agent.__class__.__name__: i 
            for i, agent in enumerate(agents)
        }
        self.idx_to_agent = {v: k for k, v in self.agent_to_idx.items()}
        
        # State: [query_embedding (768), độ dài context (1), 
        #         các agent trước (4), lịch sử satisfaction (5)]
        self.state_dim = 768 + 1 + len(agents) + 5
        self.action_dim = len(agents)
        
        # Networks
        self.policy = PolicyNetwork(
            self.state_dim, 
            self.action_dim
        ).to(device)
        
        self.value = ValueNetwork(
            self.state_dim
        ).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), 
            lr=learning_rate
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training stats
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0,
            "avg_reward": 0,
            "policy_loss": 0,
            "value_loss": 0
        }
        
        logger.info(f"RL Orchestrator initialized with {len(agents)} agents")
        logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
    
    async def _encode_state(
        self,
        query: str,
        conversation_history: List[Message],
        previous_agents: List[str],
        satisfaction_scores: List[float]
    ) -> torch.Tensor:
        """
        Encode current state
        
        Returns:
            State tensor of shape (state_dim,)
        """
        # Query embedding
        query_emb = await self.embedding_service.embed_text(query)
        query_emb = np.array(query_emb)
        
        # Context length
        context_length = min(len(conversation_history) / 20, 1.0)  # Normalize
        
        # Previous agents one-hot
        agent_one_hot = np.zeros(len(self.agents))
        for agent_name in previous_agents[-3:]:  # Last 3 agents
            if agent_name in self.agent_to_idx:
                agent_one_hot[self.agent_to_idx[agent_name]] = 1
        
        # Satisfaction history (last 5)
        satisfaction = np.array(satisfaction_scores[-5:])
        if len(satisfaction) < 5:
            satisfaction = np.pad(satisfaction, (0, 5 - len(satisfaction)))
        
        # Concatenate
        state = np.concatenate([
            query_emb,
            [context_length],
            agent_one_hot,
            satisfaction
        ])
        
        return torch.FloatTensor(state).to(self.device)
    
    async def select_agent(
        self,
        query: str,
        conversation_history: List[Message] = None,
        previous_agents: List[str] = None,
        satisfaction_scores: List[float] = None,
        explore: bool = True
    ) -> Tuple[BaseAgent, float]:
        """
        Select optimal agent using learned policy
        
        Args:
            query: User query
            conversation_history: Previous messages
            previous_agents: Previously selected agents
            satisfaction_scores: User satisfaction scores
            explore: Whether to explore (training) or exploit (inference)
            
        Returns:
            (selected_agent, confidence)
        """
        conversation_history = conversation_history or []
        previous_agents = previous_agents or []
        satisfaction_scores = satisfaction_scores or []
        
        # Encode state
        state = await self._encode_state(
            query,
            conversation_history,
            previous_agents,
            satisfaction_scores
        )
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.policy(state)
        
        # Select action
        if explore and np.random.random() < 0.1:  # 10% exploration
            action_idx = np.random.randint(self.action_dim)
            confidence = 1.0 / self.action_dim
        else:
            # Sample from distribution
            dist = Categorical(action_probs)
            action_idx = dist.sample().item()
            confidence = action_probs[action_idx].item()
        
        selected_agent = self.agents[action_idx]
        
        logger.info(
            f"Selected agent: {selected_agent.__class__.__name__} "
            f"(confidence: {confidence:.3f})"
        )
        
        return selected_agent, confidence
    
    async def learn_from_feedback(
        self,
        query: str,
        conversation_history: List[Message],
        previous_agents: List[str],
        satisfaction_scores: List[float],
        selected_agent: str,
        user_feedback: float,
        next_query: Optional[str] = None
    ):
        """
        Learn from user feedback
        
        Args:
            query: Original query
            conversation_history: Conversation context
            previous_agents: Previous agent selections
            satisfaction_scores: Historical satisfaction
            selected_agent: Agent that was selected
            user_feedback: User satisfaction (-1 to +1)
            next_query: Next query (if any) for next state
        """
        # Encode state
        state = await self._encode_state(
            query,
            conversation_history,
            previous_agents,
            satisfaction_scores
        )
        
        # Get action
        action_idx = self.agent_to_idx[selected_agent]
        
        # Compute reward
        reward = self._compute_reward(
            user_feedback,
            selected_agent,
            conversation_history
        )
        
        # Encode next state
        if next_query:
            new_satisfaction = satisfaction_scores + [user_feedback]
            new_previous = previous_agents + [selected_agent]
            next_state = await self._encode_state(
                next_query,
                conversation_history,
                new_previous,
                new_satisfaction
            )
            done = False
        else:
            next_state = state
            done = True
        
        # Store experience
        experience = Experience(
            state=state.cpu().numpy(),
            action=action_idx,
            reward=reward,
            next_state=next_state.cpu().numpy(),
            done=done
        )
        self.replay_buffer.push(experience)
        
        # Update training stats
        self.training_stats["total_reward"] += reward
        
        logger.info(
            f"Feedback received: {user_feedback:.2f}, Reward: {reward:.2f}"
        )
    
    def _compute_reward(
        self,
        user_feedback: float,
        selected_agent: str,
        conversation_history: List[Message]
    ) -> float:
        """
        Compute reward from user feedback
        
        Reward components:
        - User explicit feedback: -1 to +1
        - Conversation efficiency: -0.2 if too many back-and-forth
        - Safety bonus: +0.3 if safety-critical query handled well
        """
        reward = user_feedback
        
        # Efficiency penalty
        if len(conversation_history) > 10:
            reward -= 0.2
        
        # Safety bonus
        if "SafetyAgent" in selected_agent:
            # Check if query had safety keywords
            last_msg = conversation_history[-1] if conversation_history else None
            if last_msg and any(
                kw in last_msg.content.lower() 
                for kw in ["tác dụng phụ", "nguy hiểm", "cảnh báo", "chống chỉ định"]
            ):
                reward += 0.3
        
        return reward
    
    async def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform one training step using PPO
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor(
            np.array([exp.state for exp in batch])
        ).to(self.device)
        
        actions = torch.LongTensor(
            [exp.action for exp in batch]
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            [exp.reward for exp in batch]
        ).to(self.device)
        
        next_states = torch.FloatTensor(
            np.array([exp.next_state for exp in batch])
        ).to(self.device)
        
        dones = torch.FloatTensor(
            [exp.done for exp in batch]
        ).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values
        
        # Old policy probabilities
        with torch.no_grad():
            old_probs = self.policy(states)
            old_action_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # PPO update
        for _ in range(10):  # PPO epochs
            # Current policy
            curr_probs = self.policy(states)
            curr_action_probs = curr_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Ratio
            ratio = curr_action_probs / (old_action_probs + 1e-10)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
        
        # Value network update
        values = self.value(states).squeeze()
        value_loss = nn.MSELoss()(values, td_targets)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.value_optimizer.step()
        
        # Update stats
        self.training_stats["episodes"] += 1
        self.training_stats["policy_loss"] = policy_loss.item()
        self.training_stats["value_loss"] = value_loss.item()
        self.training_stats["avg_reward"] = (
            self.training_stats["total_reward"] / self.training_stats["episodes"]
        )
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "avg_reward": self.training_stats["avg_reward"]
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "training_stats": self.training_stats
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.training_stats = checkpoint["training_stats"]
        logger.info(f"Checkpoint loaded from {path}")


# Example usage
if __name__ == "__main__":
    import asyncio
    from backend.agents.drug_info_agent import DrugInfoAgent
    from backend.agents.interaction_agent import InteractionAgent
    from backend.agents.dosage_agent import DosageAgent
    from backend.agents.safety_agent import SafetyAgent
    from backend.services.ai_adapter import ai_adapter
    
    async def test_rl_orchestrator():
        # Initialize agents
        agents = [
            DrugInfoAgent(),
            InteractionAgent(),
            DosageAgent(),
            SafetyAgent()
        ]
        
        # Initialize RL orchestrator
        rl_orch = RLAgentOrchestrator(
            agents=agents,
            embedding_service=ai_adapter.embedding_service
        )
        
        # Test selection
        query = "Paracetamol có tác dụng phụ gì?"
        agent, confidence = await rl_orch.select_agent(query)
        
        print(f"Selected: {agent.__class__.__name__}")
        print(f"Confidence: {confidence:.3f}")
        
        # Simulate feedback
        await rl_orch.learn_from_feedback(
            query=query,
            conversation_history=[],
            previous_agents=[],
            satisfaction_scores=[],
            selected_agent=agent.__class__.__name__,
            user_feedback=0.8  # Positive feedback
        )
        
        # Train
        if len(rl_orch.replay_buffer) >= 64:
            metrics = await rl_orch.train_step()
            print(f"Training metrics: {metrics}")
    
    asyncio.run(test_rl_orchestrator())
