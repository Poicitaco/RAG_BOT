"""
Streamlit Frontend cho Hệ thống Trợ lý AI Dược phẩm
"""
import streamlit as st
import requests
import json
from typing import Dict, Any
import uuid

# Cấu hình
API_BASE_URL = "http://localhost:8000"

# Cấu hình trang
st.set_page_config(
    page_title="Trợ lý AI Dược phẩm",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}
.assistant-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
.warning-box {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# Khởi tạo session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []


def call_chat_api(message: str) -> Dict[str, Any]:
    """Gọi API chat"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat/",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Lỗi kết nối API: {str(e)}")
        return None


def main():
    """Ứng dụng chính"""
    
    # Header
    st.markdown('<h1 class="main-header"> Trợ lý AI Dược phẩm</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header(" Thông tin")
        
        st.markdown("""
        ### Trợ lý AI có thể giúp bạn:
        -  Tra cứu thông tin thuốc
        -  Kiểm tra tương tác thuốc
        -  Tư vấn liều lượng
        -  Cảnh báo an toàn
        
        ### Ví dụ câu hỏi:
        - "Paracetamol 500mg dùng như thế nào?"
        - "Tôi có thể dùng Aspirin và Ibuprofen cùng lúc không?"
        - "Liều lượng Amoxicillin cho trẻ 5 tuổi?"
        """)
        
        st.divider()
        
        # Lưu ý y tế quan trọng
        st.warning("""
         **LƯU Ý QUAN TRỌNG**
        
        Thông tin chỉ mang tính chất tham khảo. 
        Luôn tham khảo bác sĩ/dược sĩ trước khi dùng thuốc.
        """)
        
        st.divider()
        
        if st.button(" Xóa lịch sử chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Giao diện chat chính
    chat_container = st.container()
    
    with chat_container:
        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(
                    f'<div class="chat-message user-message"> <strong>Bạn:</strong><br/>{content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"> <strong>Trợ lý:</strong><br/>{content}</div>',
                    unsafe_allow_html=True
                )
                
                # Hiển thị cảnh báo nếu có
                if "warnings" in message and message["warnings"]:
                    for warning in message["warnings"]:
                        st.markdown(
                            f'<div class="warning-box">{warning}</div>',
                            unsafe_allow_html=True
                        )
                
                # Hiển thị gợi ý
                if "suggestions" in message and message["suggestions"]:
                    st.markdown("** Gợi ý:**")
                    cols = st.columns(min(3, len(message["suggestions"])))
                    for idx, suggestion in enumerate(message["suggestions"][:3]):
                        with cols[idx]:
                            if st.button(suggestion, key=f"sugg_{len(st.session_state.messages)}_{idx}"):
                                process_message(suggestion)
                                st.rerun()
    
    # Ô nhập liệu chat
    st.divider()
    
    # Tạo cột cho ô nhập và nút
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Nhập câu hỏi của bạn:",
            key="user_input",
            placeholder="Ví dụ: Paracetamol dùng như thế nào?"
        )
    
    with col2:
        send_button = st.button(" Gửi", use_container_width=True)
    
    # Xử lý tin nhắn khi gửi
    if send_button and user_input:
        process_message(user_input)
        st.rerun()
    
    # Các hành động nhanh
    st.divider()
    st.markdown("###  Hành động nhanh")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(" Tra cứu thuốc", use_container_width=True):
            process_message("Tôi muốn tra cứu thông tin về một loại thuốc")
            st.rerun()
    
    with col2:
        if st.button(" Kiểm tra tương tác", use_container_width=True):
            process_message("Tôi muốn kiểm tra tương tác giữa các loại thuốc")
            st.rerun()
    
    with col3:
        if st.button(" Tư vấn liều lượng", use_container_width=True):
            process_message("Tôi muốn tư vấn về liều lượng sử dụng thuốc")
            st.rerun()
    
    with col4:
        if st.button(" An toàn thuốc", use_container_width=True):
            process_message("Cho tôi biết về an toàn khi sử dụng thuốc")
            st.rerun()


def process_message(message: str):
    """Xử lý tin nhắn người dùng và lấy phản hồi"""
    # Thêm tin nhắn người dùng
    st.session_state.messages.append({
        "role": "user",
        "content": message
    })
    
    # Hiển thị chỉ báo đang xử lý
    with st.spinner("Đang xử lý..."):
        # Gọi API
        response = call_chat_api(message)
    
    if response:
        # Thêm phản hồi của trợ lý
        assistant_message = {
            "role": "assistant",
            "content": response.get("message", ""),
            "warnings": response.get("warnings", []),
            "suggestions": response.get("suggestions", []),
            "agent_type": response.get("agent_type", "")
        }
        st.session_state.messages.append(assistant_message)


# Các trang bổ sung (sử dụng tabs)
def show_about():
    """Hiển thị trang giới thiệu"""
    st.header(" Về Trợ lý AI Dược phẩm")
    
    st.markdown("""
    ## Giới thiệu
    
    Trợ lý AI Dược phẩm là hệ thống AI thông minh sử dụng công nghệ RAG 
    (Retrieval-Augmented Generation) và Multi-Agent để hỗ trợ người dân 
    tra cứu thông tin về thuốc và sức khỏe.
    
    ## Tính năng chính
    
    - ** Tra cứu thông tin thuốc**: Cung cấp thông tin chi tiết về thuốc
    - ** Kiểm tra tương tác**: Phát hiện tương tác giữa các loại thuốc
    - ** Tư vấn liều lượng**: Hướng dẫn sử dụng liều lượng phù hợp
    - ** Giám sát an toàn**: Cảnh báo tác dụng phụ và chống chỉ định
    
    ## Công nghệ
    
    - **Backend**: FastAPI, LangChain, OpenAI GPT-4
    - **RAG System**: ChromaDB, Semantic Search
    - **Multi-Agent**: Các agent chuyên biệt cho các nhiệm vụ khác nhau
    - **Frontend**: Streamlit
    
    ## Disclaimer
    
     Thông tin chỉ mang tính chất tham khảo. Không thay thế tư vấn y tế chuyên nghiệp.
    """)


# Thực thi chính
if __name__ == "__main__":
    main()
