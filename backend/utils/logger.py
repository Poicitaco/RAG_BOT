"""
Logging utility for the application - Tiện ích logging cho ứng dụng
"""
import sys
from loguru import logger
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """Thiết lập và cấu hình logger ứng dụng"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "./logs"):
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Cấu hình logger với file và console handlers"""
        # Remove default handler
        logger.remove()
        
        # Console handler with color
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=self.log_level,
            colorize=True,
        )
        
        # File handler cho tất cả logs
        logger.add(
            self.log_dir / "app.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )
        
        # File handler chỉ cho errors
        logger.add(
            self.log_dir / "error.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            level="ERROR",
            rotation="10 MB",
            retention="90 days",
            compression="zip",
        )
        
        # File handler cho API requests
        logger.add(
            self.log_dir / "api.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            filter=lambda record: "API" in record["extra"].get("type", ""),
        )
    
    @staticmethod
    def get_logger(name: Optional[str] = None):
        """Lấy logger instance"""
        if name:
            return logger.bind(name=name)
        return logger


# Initialize logger
log_setup = LoggerSetup()
app_logger = log_setup.get_logger("PharmaAI")


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: Optional[str] = None
) -> None:
    """Ghi log chi tiết API request"""
    logger.bind(type="API").info(
        f"{method} {path} - Status: {status_code} - "
        f"Duration: {duration:.3f}s - User: {user_id or 'anonymous'}"
    )


def log_agent_action(agent_name: str, action: str, details: str = "") -> None:
    """Log agent actions"""
    logger.bind(type="AGENT").info(
        f"Agent: {agent_name} | Action: {action} | {details}"
    )


def log_rag_query(query: str, num_results: int, duration: float) -> None:
    """Log RAG query details"""
    logger.bind(type="RAG").info(
        f"Query: {query[:100]}... | Results: {num_results} | Duration: {duration:.3f}s"
    )
