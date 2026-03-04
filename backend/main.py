"""
Ứng dụng chính FastAPI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from backend.config import settings
from backend.utils import app_logger
from backend.api.routes import chat, drug, image, voice


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Sự kiện vòng đời ứng dụng"""
    # Khởi động
    app_logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    app_logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    yield
    
    # Tắt ứng dụng
    app_logger.info("Đang tắt ứng dụng")


# Tạo ứng dụng FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Hệ thống AI Agent đa phương tiện hỗ trợ dược phẩm",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware đo thời gian xử lý request


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Kiểm tra sức khỏe hệ thống
@app.get("/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


# Endpoint gốc
@app.get("/")
async def root():
    """Endpoint gốc"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# Thêm các router
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(drug.router, prefix="/api/v1/drug", tags=["Drug"])
app.include_router(image.router, prefix="/api/v1/image", tags=["Image"])
app.include_router(voice.router, prefix="/api/v1/voice", tags=["Voice"])


# Xử lý lỗi
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    app_logger.error(f"Lỗi không được xử lý: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Lỗi máy chủ nội bộ",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
