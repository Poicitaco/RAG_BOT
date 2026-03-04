"""
MARL Coordinator - Multi-Agent Reinforcement Learning
Phối hợp nhiều RL agent sử dụng thuật toán QMIX
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import deque
from loguru import logger


@dataclass
class MARLExperience:
    """Kinh nghiệm cho huấn luyện MARL"""
    state: np.ndarray  # Global state
    agent_observations: List[np.ndarray]  # Per-agent observations
    agent_actions: List[int]  # Actions taken by each agent
    rewards: List[float]  # Per-agent rewards
    next_state: np.ndarray
    next_agent_observations: List[np.ndarray]
    done: bool


class AgentQNetwork(nn.Module):
    """
    Mạng Q cho từng agent riêng lẻ
    
    Mỗi agent có mạng Q riêng ước tính giá trị hành động
    dựa trên quan sát cục bộ
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Khởi tạo mạng Q agent
        
        Args:
            obs_dim: Số chiều quan sát
            action_dim: Số lượng hành động
            hidden_dim: Số chiều lớp ẩn
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            observation: (batch_size, obs_dim)
            
        Returns:
            q_values: (batch_size, action_dim)
        """
        return self.network(observation)


class QMixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    
    Combines individual agent Q-values into team Q-value
    Ensures that joint action selection is consistent with
    individual agent greedy action selection (monotonicity)
    
    Paper: "QMIX: Monotonic Value Function Factorisation for 
    Decentralised Multi-Agent Reinforcement Learning" (2018)
    """
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 64,
        hypernet_hidden: int = 64
    ):
        """
        Khởi tạo mạng QMIX
        
        Args:
            num_agents: Số lượng agents
            state_dim: Số chiều state toàn cục
            embed_dim: Số chiều embedding
            hypernet_hidden: Số chiều ẩn hypernetwork
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # Hypernetwork cho trọng số mixing (layer 1)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, num_agents * embed_dim)
        )
        
        # Hypernetwork cho bias mixing (layer 1)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        
        # Hypernetwork cho trọng số mixing (layer 2)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden),
            nn.ReLU(),
            nn.Linear(hypernet_hidden, embed_dim)
        )
        
        # Hypernetwork cho bias mixing (layer 2)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Kết hợp giá trị Q của các agent thành giá trị Q nhóm
        
        Args:
            agent_q_values: (batch_size, num_agents) - Giá trị Q từ mỗi agent
            state: (batch_size, state_dim) - State toàn cục
            
        Returns:
            q_total: (batch_size, 1) - Giá trị Q nhóm
        """
        batch_size = agent_q_values.size(0)
        
        # Tạo trọng số và bias mixing từ state
        w1 = torch.abs(self.hyper_w1(state))  # Trọng số dương cho tính đơn điệu
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        
        # Lớp mixing thứ nhất
        # agent_q_values: (batch_size, num_agents, 1)
        agent_q_values = agent_q_values.unsqueeze(2)
        
        # hidden: (batch_size, 1, embed_dim)
        hidden = F.elu(torch.bmm(agent_q_values.transpose(1, 2), w1) + b1)
        
        # Lớp mixing thứ hai
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        # q_total: (batch_size, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.squeeze()


class MARLCoordinator:
    """
    Multi-Agent RL Coordinator using QMIX
    
    Coordinates 4 pharmaceutical agents:
    1. Drug Info Agent
    2. Interaction Agent
    3. Dosage Agent
    4. Safety Agent
    
    Features:
    - Decentralized execution (each agent acts independently)
    - Centralized training (mixer learns coordination)
    - Credit assignment (which agent contributed to success)
    - Joint optimization (maximize team reward)
    """
    
    def __init__(
        self,
        num_agents: int = 4,
        obs_dim: int = 768,  # Embedding dimension
        action_dim: int = 2,  # Use agent or not
        state_dim: int = 800,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Khởi tạo MARL coordinator
        
        Args:
            num_agents: Số lượng agents (4 cho hệ thống của chúng ta)
            obs_dim: Số chiều quan sát cho mỗi agent
            action_dim: Không gian hành động cho mỗi agent
            state_dim: Số chiều state toàn cục
            learning_rate: Tốc độ học
            gamma: Hệ số chiết khấu
            device: 'cuda' hoặc 'cpu'
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.device = device
        
        # Create agent Q-networks
        self.agent_networks = nn.ModuleList([
            AgentQNetwork(obs_dim, action_dim).to(device)
            for _ in range(num_agents)
        ])
        
        # Create mixing network
        self.mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim
        ).to(device)
        
        # Target networks (for stable learning)
        self.target_agent_networks = nn.ModuleList([
            AgentQNetwork(obs_dim, action_dim).to(device)
            for _ in range(num_agents)
        ])
        
        self.target_mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim
        ).to(device)
        
        # Copy weights to target networks
        self._update_target_networks()
        
        # Optimizer
        params = list(self.mixer.parameters())
        for agent_net in self.agent_networks:
            params += list(agent_net.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        
        # Statistics
        self.training_step = 0
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        logger.info(
            f"MARL Coordinator initialized: "
            f"{num_agents} agents, "
            f"obs_dim={obs_dim}, "
            f"device={device}"
        )
    
    def select_actions(
        self,
        observations: List[np.ndarray],
        explore: bool = True
    ) -> List[int]:
        """
        Select actions for all agents
        
        Args:
            observations: List of observations (one per agent)
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            actions: List of actions (one per agent)
        """
        actions = []
        
        for i, obs in enumerate(observations):
            # Epsilon-greedy exploration
            if explore and np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                # Greedy action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.agent_networks[i](obs_tensor)
                    action = q_values.argmax(dim=1).item()
            
            actions.append(action)
        
        return actions
    
    def add_experience(self, experience: MARLExperience):
        """Add experience to replay buffer"""
        self.replay_buffer.append(experience)
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch_size: Batch size
            
        Returns:
            metrics: Training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        # Sample batch
        batch = np.random.choice(self.replay_buffer, size=batch_size, replace=False)
        
        # Prepare tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        
        # Agent observations and actions
        agent_obs = [
            torch.FloatTensor([exp.agent_observations[i] for exp in batch]).to(self.device)
            for i in range(self.num_agents)
        ]
        
        next_agent_obs = [
            torch.FloatTensor([exp.next_agent_observations[i] for exp in batch]).to(self.device)
            for i in range(self.num_agents)
        ]
        
        actions = torch.LongTensor([exp.agent_actions for exp in batch]).to(self.device)
        
        rewards = torch.FloatTensor([sum(exp.rewards) for exp in batch]).unsqueeze(1).to(self.device)
        
        dones = torch.FloatTensor([exp.done for exp in batch]).unsqueeze(1).to(self.device)
        
        # Compute current Q-values
        agent_q_values = []
        for i in range(self.num_agents):
            q_vals = self.agent_networks[i](agent_obs[i])
            # Select Q-values for taken actions
            q_vals = q_vals.gather(1, actions[:, i].unsqueeze(1))
            agent_q_values.append(q_vals)
        
        agent_q_values = torch.cat(agent_q_values, dim=1)  # (batch_size, num_agents)
        
        # Mix Q-values
        q_total = self.mixer(agent_q_values, states).unsqueeze(1)  # (batch_size, 1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_agent_q_values = []
            for i in range(self.num_agents):
                next_q_vals = self.target_agent_networks[i](next_agent_obs[i])
                next_q_vals = next_q_vals.max(dim=1)[0].unsqueeze(1)
                next_agent_q_values.append(next_q_vals)
            
            next_agent_q_values = torch.cat(next_agent_q_values, dim=1)
            
            next_q_total = self.target_mixer(next_agent_q_values, next_states).unsqueeze(1)
            
            target_q_total = rewards + self.gamma * next_q_total * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(q_total, target_q_total)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), max_norm=10.0)
        for agent_net in self.agent_networks:
            torch.nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # Update target networks periodically
        self.training_step += 1
        if self.training_step % 100 == 0:
            self._update_target_networks()
        
        return {
            "loss": loss.item(),
            "q_value": q_total.mean().item(),
            "epsilon": self.epsilon,
            "training_step": self.training_step
        }
    
    def _update_target_networks(self):
        """Update target networks with current network weights"""
        for target_net, net in zip(self.target_agent_networks, self.agent_networks):
            target_net.load_state_dict(net.state_dict())
        
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            "agent_networks": [net.state_dict() for net in self.agent_networks],
            "mixer": self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "epsilon": self.epsilon
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(checkpoint["agent_networks"][i])
        
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
        self.epsilon = checkpoint["epsilon"]
        
        self._update_target_networks()
        
        logger.info(f"Checkpoint loaded: {path}")


# Demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMO BỘ ĐIỀU PHỐI MARL")
    print("="*70)
    
    # Khởi tạo coordinator
    coordinator = MARLCoordinator(
        num_agents=4,
        obs_dim=768,
        action_dim=2,
        state_dim=800
    )
    
    print(f"\nĐã khởi tạo {coordinator.num_agents} agents với QMIX")
    
    # Mô phỏng huấn luyện
    print("\nMô phỏng quá trình ra quyết định có phối hợp...")
    
    for episode in range(10):
        # Quan sát ngẫu nhiên (trong hệ thống thực tế, đây là embeddings)
        observations = [np.random.randn(768) for _ in range(4)]
        
        # Chọn hành động
        actions = coordinator.select_actions(observations, explore=True)
        
        print(f"\nEpisode {episode+1}:")
        print(f"  Actions: {actions}")
        print(f"  Epsilon: {coordinator.epsilon:.3f}")
        
        # Mô phỏng kinh nghiệm
        experience = MARLExperience(
            state=np.random.randn(800),
            agent_observations=observations,
            agent_actions=actions,
            rewards=[np.random.random() for _ in range(4)],
            next_state=np.random.randn(800),
            next_agent_observations=[np.random.randn(768) for _ in range(4)],
            done=False
        )
        
        coordinator.add_experience(experience)
        
        # Huấn luyện nếu đủ dữ liệu
        if len(coordinator.replay_buffer) >= 5:
            metrics = coordinator.train_step(batch_size=5)
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Q-value: {metrics['q_value']:.4f}")
    
    print("\nDemo hoàn thành!")
    print("\nMARL cho phép:")
    print("  - Phối hợp giữa các agents")
    print("  - Phân bổ credit (agent nào giúp được?)")
    print("  - Tối ưu hóa chung")
    print("  - Hành vi làm việc nhóm tự nhiên")
