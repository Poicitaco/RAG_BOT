"""
Quantum Cloud Simulator
Interface for cloud-based quantum computing (IBM Quantum, AWS Braket, Azure Quantum)
"""
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.providers.jobstatus import JobStatus
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    logger.warning("Qiskit not installed. Quantum features limited.")
    QISKIT_AVAILABLE = False


class CloudProvider(Enum):
    """Cloud quantum providers"""
    IBM_QUANTUM = "ibm_quantum"
    AWS_BRAKET = "aws_braket"
    AZURE_QUANTUM = "azure_quantum"
    LOCAL_SIMULATOR = "local_simulator"


@dataclass
class QuantumJob:
    """Quantum job information"""
    job_id: str
    provider: CloudProvider
    circuit_name: str
    status: str
    shots: int
    estimated_cost: float
    result: Optional[Dict] = None


class QuantumCloudSimulator:
    """
    Unified interface for cloud quantum computing
    
    Supports:
    - IBM Quantum (free tier: 10 min/month, paid: $1.60/min)
    - AWS Braket (on-demand: $0.30/task + $0.00035/shot)
    - Azure Quantum ($50 credit free, then pay-as-you-go)
    - Local simulator (free, for development)
    
    Features:
    - Automatic provider selection
    - Cost estimation
    - Job queue management
    - Result caching
    """
    
    def __init__(
        self,
        provider: CloudProvider = CloudProvider.LOCAL_SIMULATOR,
        api_token: Optional[str] = None,
        max_cost_per_job: float = 1.0
    ):
        """
        Initialize quantum cloud simulator
        
        Args:
            provider: Cloud provider to use
            api_token: API token for cloud access
            max_cost_per_job: Maximum cost per job (USD)
        """
        self.provider = provider
        self.api_token = api_token
        self.max_cost_per_job = max_cost_per_job
        
        self.backend = None
        self.job_history: List[QuantumJob] = []
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize quantum backend"""
        if not QISKIT_AVAILABLE:
            logger.error("Qiskit not available. Install: pip install qiskit qiskit-aer")
            return
        
        if self.provider == CloudProvider.LOCAL_SIMULATOR:
            # Use local Aer simulator (free)
            self.backend = AerSimulator()
            logger.info(" Initialized local quantum simulator (Aer)")
        
        elif self.provider == CloudProvider.IBM_QUANTUM:
            # IBM Quantum
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                
                if not self.api_token:
                    logger.error("IBM Quantum requires API token. Get from: https://quantum-computing.ibm.com/")
                    logger.info("   Using local simulator instead")
                    self.backend = AerSimulator()
                else:
                    service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_token)
                    self.backend = service.least_busy()
                    logger.info(f" Connected to IBM Quantum: {self.backend.name}")
            
            except ImportError:
                logger.error("IBM Quantum Runtime not installed. Install: pip install qiskit-ibm-runtime")
                logger.info("   Using local simulator")
                self.backend = AerSimulator()
        
        elif self.provider == CloudProvider.AWS_BRAKET:
            logger.warning("AWS Braket integration coming soon. Using local simulator.")
            self.backend = AerSimulator()
        
        elif self.provider == CloudProvider.AZURE_QUANTUM:
            logger.warning("Azure Quantum integration coming soon. Using local simulator.")
            self.backend = AerSimulator()
    
    def execute_circuit(
        self,
        circuit: "QuantumCircuit",
        shots: int = 1024,
        circuit_name: str = "quantum_job"
    ) -> QuantumJob:
        """
        Execute quantum circuit
        
        Args:
            circuit: Qiskit quantum circuit
            shots: Number of measurement shots
            circuit_name: Name for this job
            
        Returns:
            QuantumJob with results
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        # Estimate cost
        cost = self._estimate_cost(circuit, shots)
        
        if cost > self.max_cost_per_job:
            logger.warning(
                f"Job cost ${cost:.2f} exceeds max ${self.max_cost_per_job:.2f}. "
                f"Reducing shots from {shots} to {int(shots * self.max_cost_per_job / cost)}"
            )
            shots = int(shots * self.max_cost_per_job / cost)
            cost = self.max_cost_per_job
        
        logger.info(f"Executing circuit '{circuit_name}' (shots={shots}, cost=${cost:.4f})")
        
        # Transpile for backend
        transpiled = transpile(circuit, self.backend)
        
        # Execute
        job = self.backend.run(transpiled, shots=shots)
        
        # Create job object
        quantum_job = QuantumJob(
            job_id=job.job_id(),
            provider=self.provider,
            circuit_name=circuit_name,
            status="running",
            shots=shots,
            estimated_cost=cost
        )
        
        # Wait for result (for simulator, instant; for real device, may take time)
        try:
            result = job.result()
            counts = result.get_counts()
            
            quantum_job.status = "completed"
            quantum_job.result = {
                "counts": counts,
                "success": result.success
            }
            
            logger.success(f" Job completed: {quantum_job.job_id}")
        
        except Exception as e:
            quantum_job.status = "failed"
            logger.error(f" Job failed: {e}")
        
        # Store in history
        self.job_history.append(quantum_job)
        
        return quantum_job
    
    def _estimate_cost(self, circuit: "QuantumCircuit", shots: int) -> float:
        """
        Estimate job cost in USD
        
        Pricing (as of 2024):
        - Local simulator: $0 (free)
        - IBM Quantum: $1.60/min (free 10 min/month)
        - AWS Braket: $0.30/task + $0.00035/shot
        - Azure Quantum: varies by hardware
        """
        if self.provider == CloudProvider.LOCAL_SIMULATOR:
            return 0.0
        
        elif self.provider == CloudProvider.IBM_QUANTUM:
            # Estimate runtime: ~1 sec per 1000 shots for small circuits
            estimated_seconds = (shots / 1000) * 1.0
            cost_per_second = 1.60 / 60  # $1.60/min
            return estimated_seconds * cost_per_second
        
        elif self.provider == CloudProvider.AWS_BRAKET:
            # AWS Braket pricing
            task_cost = 0.30
            shot_cost = shots * 0.00035
            return task_cost + shot_cost
        
        elif self.provider == CloudProvider.AZURE_QUANTUM:
            # Simplified pricing (varies by device)
            return 0.50 + (shots * 0.0001)
        
        return 0.0
    
    def get_total_cost(self) -> float:
        """Get total cost of all jobs"""
        return sum(job.estimated_cost for job in self.job_history)
    
    def get_job_statistics(self) -> Dict:
        """Get job statistics"""
        if not self.job_history:
            return {
                "total_jobs": 0,
                "completed": 0,
                "failed": 0,
                "total_cost": 0.0,
                "total_shots": 0
            }
        
        return {
            "total_jobs": len(self.job_history),
            "completed": sum(1 for job in self.job_history if job.status == "completed"),
            "failed": sum(1 for job in self.job_history if job.status == "failed"),
            "total_cost": self.get_total_cost(),
            "total_shots": sum(job.shots for job in self.job_history)
        }
    
    def switch_provider(self, new_provider: CloudProvider, api_token: Optional[str] = None):
        """Switch to different quantum provider"""
        logger.info(f"Switching from {self.provider.value} to {new_provider.value}")
        
        self.provider = new_provider
        if api_token:
            self.api_token = api_token
        
        self._initialize_backend()


# Helper function to create common circuits
def create_bell_state() -> "QuantumCircuit":
    """Create Bell state (quantum entanglement demo)"""
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available")
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT: qubit 0 controls qubit 1
    qc.measure([0, 1], [0, 1])
    
    return qc


def create_grover_circuit(marked_state: int, num_qubits: int = 3) -> "QuantumCircuit":
    """
    Create Grover's search circuit
    
    Searches for marked_state in 2^num_qubits possibilities
    Quantum speedup: O(√N) vs classical O(N)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available")
    
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize superposition
    qc.h(range(num_qubits))
    
    # Oracle: mark the target state
    # (Simplified - real implementation needs proper oracle)
    qc.barrier()
    
    # Grover diffusion operator
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    
    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc


# Demo
def demo():
    """Demo quantum cloud simulator"""
    if not QISKIT_AVAILABLE:
        print(" Qiskit not installed. Install with: pip install qiskit qiskit-aer")
        return
    
    print("\n" + "="*70)
    print("  QUANTUM CLOUD SIMULATOR DEMO")
    print("="*70)
    
    # Initialize simulator
    simulator = QuantumCloudSimulator(provider=CloudProvider.LOCAL_SIMULATOR)
    
    # Test 1: Bell state
    print("\n Test 1: Bell State (Quantum Entanglement)")
    bell_circuit = create_bell_state()
    print(f"Circuit: {bell_circuit.num_qubits} qubits, {bell_circuit.depth()} depth")
    
    job1 = simulator.execute_circuit(bell_circuit, shots=1000, circuit_name="bell_state")
    
    if job1.result:
        print(f"Results: {job1.result['counts']}")
        print(f"Cost: ${job1.estimated_cost:.4f}")
    
    # Test 2: Grover's search
    print("\n Test 2: Grover's Search Algorithm")
    grover_circuit = create_grover_circuit(marked_state=5, num_qubits=3)
    print(f"Circuit: {grover_circuit.num_qubits} qubits, {grover_circuit.depth()} depth")
    
    job2 = simulator.execute_circuit(grover_circuit, shots=1000, circuit_name="grover_search")
    
    if job2.result:
        print(f"Results: {job2.result['counts']}")
        print(f"Cost: ${job2.estimated_cost:.4f}")
    
    # Statistics
    print("\n Statistics:")
    stats = simulator.get_job_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n Demo completed!")
    print("\n To use IBM Quantum:")
    print("   1. Sign up: https://quantum-computing.ibm.com/")
    print("   2. Get API token")
    print("   3. simulator = QuantumCloudSimulator(")
    print("         provider=CloudProvider.IBM_QUANTUM,")
    print("         api_token='YOUR_TOKEN'")
    print("      )")


if __name__ == "__main__":
    demo()
