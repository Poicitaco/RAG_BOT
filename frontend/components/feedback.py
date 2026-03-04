"""
Feedback UI Components cho Streamlit
Thu thập phản hồi người dùng để huấn luyện RLHF
"""
import streamlit as st
from typing import Optional, Callable
from datetime import datetime
import json


class FeedbackWidget:
    """
    Widget feedback tái sử dụng để thu thập đánh giá người dùng
    
    Tính năng:
    - Nút like/dislike (👍👎)
    - Đánh giá 5 sao
    - Phản hồi văn bản (tùy chọn)
    - Theo dõi session
    """
    
    def __init__(self, session_key: str = "feedback"):
        """
        Khởi tạo feedback widget
        
        Args:
            session_key: Key session state để theo dõi
        """
        self.session_key = session_key
        
        # Khởi tạo session state
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                "history": [],
                "total_positive": 0,
                "total_negative": 0
            }
    
    def render_thumbs(
        self,
        message_id: str,
        on_feedback: Optional[Callable] = None,
        key_suffix: str = ""
    ) -> Optional[int]:
        """
        Hiển thị nút like/dislike
        
        Args:
            message_id: ID duy nhất cho tin nhắn này
            on_feedback: Hàm callback(message_id, rating)
            key_suffix: Hậu tố duy nhất cho widget keys
            
        Returns:
            rating: +1 cho like, -1 cho dislike, None nếu chưa phản hồi
        """
        col1, col2, col3 = st.columns([1, 1, 8])
        
        rating = None
        
        with col1:
            if st.button("", key=f"thumbs_up_{message_id}_{key_suffix}", help="Hữu ích"):
                rating = 1
                self._record_feedback(message_id, rating, "thumbs_up")
                if on_feedback:
                    on_feedback(message_id, rating)
                st.success("Cảm ơn phản hồi! ")
        
        with col2:
            if st.button("", key=f"thumbs_down_{message_id}_{key_suffix}", help="Không hữu ích"):
                rating = -1
                self._record_feedback(message_id, rating, "thumbs_down")
                if on_feedback:
                    on_feedback(message_id, rating)
                st.warning("Cảm ơn phản hồi! Chúng tôi sẽ cải thiện. ")
        
        return rating
    
    def render_star_rating(
        self,
        message_id: str,
        on_feedback: Optional[Callable] = None,
        key_suffix: str = ""
    ) -> Optional[int]:
        """
        Hiển thị đánh giá 5 sao
        
        Args:
            message_id: ID duy nhất cho tin nhắn này
            on_feedback: Hàm callback(message_id, rating)
            key_suffix: Hậu tố duy nhất cho widget keys
            
        Returns:
            rating: 1-5 sao, hoặc None
        """
        st.write("**Đánh giá chất lượng:**")
        
        # Star buttons
        cols = st.columns(5)
        rating = None
        
        for i, col in enumerate(cols, 1):
            with col:
                if st.button("" * i, key=f"star_{i}_{message_id}_{key_suffix}"):
                    rating = i
                    # Chuyển về tháng -1 đến +1
                    normalized_rating = (rating - 3) / 2  # 1->-1, 3->0, 5->1
                    self._record_feedback(message_id, normalized_rating, f"{rating}_stars")
                    if on_feedback:
                        on_feedback(message_id, normalized_rating)
                    st.success(f"Đã đánh giá {rating} ")
        
        return rating
    
    def render_detailed_feedback(
        self,
        message_id: str,
        on_feedback: Optional[Callable] = None,
        key_suffix: str = ""
    ) -> Optional[dict]:
        """
        Hiển thị form phản hồi chi tiết
        
        Args:
            message_id: ID duy nhất cho tin nhắn này
            on_feedback: Hàm callback(message_id, feedback_dict)
            key_suffix: Hậu tố duy nhất cho widget keys
            
        Returns:
            feedback_dict: Dict với rating, categories và text
        """
        with st.expander(" Phản hồi chi tiết (tuỳ chọn)"):
            # Rating
            rating = st.slider(
                "Chất lượng tổng thể:",
                min_value=1,
                max_value=5,
                value=3,
                key=f"rating_slider_{message_id}_{key_suffix}"
            )
            
            # Categories
            st.write("**Đánh giá cụ thể:**")
            accuracy = st.checkbox(" Thông tin chính xác", key=f"acc_{message_id}_{key_suffix}")
            helpful = st.checkbox(" Hữu ích", key=f"help_{message_id}_{key_suffix}")
            clear = st.checkbox(" Dễ hiểu", key=f"clear_{message_id}_{key_suffix}")
            complete = st.checkbox(" Đầy đủ", key=f"complete_{message_id}_{key_suffix}")
            
            # Text feedback
            text_feedback = st.text_area(
                "Góp ý thêm:",
                placeholder="Bạn có góp ý gì để chúng tôi cải thiện?",
                key=f"text_{message_id}_{key_suffix}"
            )
            
            # Submit button
            if st.button("Gửi phản hồi", key=f"submit_{message_id}_{key_suffix}"):
                feedback_dict = {
                    "rating": rating,
                    "normalized_rating": (rating - 3) / 2,
                    "categories": {
                        "accuracy": accuracy,
                        "helpful": helpful,
                        "clear": clear,
                        "complete": complete
                    },
                    "text": text_feedback,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._record_feedback(
                    message_id,
                    feedback_dict["normalized_rating"],
                    "detailed",
                    extra_data=feedback_dict
                )
                
                if on_feedback:
                    on_feedback(message_id, feedback_dict)
                
                st.success(" Cảm ơn phản hồi chi tiết của bạn!")
                
                return feedback_dict
        
        return None
    
    def _record_feedback(
        self,
        message_id: str,
        rating: float,
        feedback_type: str,
        extra_data: Optional[dict] = None
    ):
        """Ghi lại phản hồi trong session state"""
        feedback_entry = {
            "message_id": message_id,
            "rating": rating,
            "type": feedback_type,
            "timestamp": datetime.now().isoformat(),
            "extra_data": extra_data
        }
        
        st.session_state[self.session_key]["history"].append(feedback_entry)
        
        # Cập nhật bộ đếm
        if rating > 0:
            st.session_state[self.session_key]["total_positive"] += 1
        elif rating < 0:
            st.session_state[self.session_key]["total_negative"] += 1
    
    def get_statistics(self) -> dict:
        """Lấy thống kê phản hồi"""
        data = st.session_state[self.session_key]
        
        if not data["history"]:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "satisfaction_rate": 0.0
            }
        
        total = len(data["history"])
        positive = data["total_positive"]
        negative = data["total_negative"]
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": positive / (positive + negative) if (positive + negative) > 0 else 0.5
        }
    
    def render_statistics(self):
        """Hiển thị thống kê phản hồi"""
        stats = self.get_statistics()
        
        st.sidebar.markdown("---")
        st.sidebar.subheader(" Phản hồi")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(" Tích cực", stats["positive"])
        with col2:
            st.metric(" Tiêu cực", stats["negative"])
        
        if stats["total"] > 0:
            st.sidebar.progress(stats["satisfaction_rate"])
            st.sidebar.caption(f"Tỷ lệ hài lòng: {stats['satisfaction_rate']:.1%}")


class ChatFeedback:
    """
    Hệ thống feedback tích hợp với giao diện chat
    
    Tự động theo dõi cặp Hỏi-Đáp và thu thập phản hồi
    """
    
    def __init__(self):
        """Khởi tạo hệ thống chat feedback"""
        self.widget = FeedbackWidget(session_key="chat_feedback")
        
        # Khởi tạo chat state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "feedback_data" not in st.session_state:
            st.session_state.feedback_data = []
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Thêm tin nhắn vào chat và trả về message ID
        
        Args:
            role: 'user' hoặc 'assistant'
            content: Nội dung tin nhắn
            metadata: Metadata bổ sung (agent, context, v.v.)
            
        Returns:
            message_id: ID duy nhất cho tin nhắn này
        """
        message_id = f"msg_{len(st.session_state.messages)}"
        
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "feedback": None
        }
        
        st.session_state.messages.append(message)
        
        return message_id
    
    def render_chat_with_feedback(self):
        """Hiển thị tin nhắn chat với feedback inline"""
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Hiển thị feedback widget cho tin nhắn assistant
                if message["role"] == "assistant" and message["feedback"] is None:
                    st.markdown("---")
                    
                    # Callback để lưu feedback
                    def on_feedback(msg_id: str, rating: float):
                        # Tìm message và cập nhật
                        for msg in st.session_state.messages:
                            if msg["id"] == msg_id:
                                msg["feedback"] = rating
                                
                                # Lưu cho backend
                                st.session_state.feedback_data.append({
                                    "message_id": msg_id,
                                    "query": st.session_state.messages[i-1]["content"] if i > 0 else "",
                                    "response": msg["content"],
                                    "rating": rating,
                                    "metadata": msg["metadata"],
                                    "timestamp": datetime.now().isoformat()
                                })
                                break
                    
                    # Hiển thị nút thumbs
                    self.widget.render_thumbs(
                        message["id"],
                        on_feedback=on_feedback,
                        key_suffix=str(i)
                    )
                
                # Hiển thị trạng thái feedback nếu đã đưa
                elif message["role"] == "assistant" and message["feedback"] is not None:
                    if message["feedback"] > 0:
                        st.caption(" Bạn đã đánh giá: Hữu ích")
                    else:
                        st.caption(" Bạn đã đánh giá: Không hữu ích")
    
    def get_feedback_for_export(self) -> list:
        """Lấy tất cả dữ liệu feedback để export/huấn luyện"""
        return st.session_state.feedback_data
    
    def export_feedback_json(self, filepath: str):
        """Xuất feedback ra file JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.feedback_data, f, ensure_ascii=False, indent=2)
    
    def render_sidebar_stats(self):
        """Hiển thị thống kê feedback trong sidebar"""
        self.widget.render_statistics()


# Ví dụ sử dụng
def demo_feedback_ui():
    """Demo các component feedback UI"""
    st.set_page_config(page_title="Feedback Demo", page_icon="")
    
    st.title(" Feedback UI Demo")
    
    # Method 1: Simple thumbs
    st.header("1. Simple Thumbs Up/Down")
    widget = FeedbackWidget()
    widget.render_thumbs("demo_message_1", key_suffix="demo1")
    
    st.markdown("---")
    
    # Method 2: Star rating
    st.header("2. Star Rating")
    widget.render_star_rating("demo_message_2", key_suffix="demo2")
    
    st.markdown("---")
    
    # Method 3: Detailed feedback
    st.header("3. Detailed Feedback")
    widget.render_detailed_feedback("demo_message_3", key_suffix="demo3")
    
    st.markdown("---")
    
    # Method 4: Chat with feedback
    st.header("4. Chat with Feedback")
    chat_feedback = ChatFeedback()
    
    # Add some demo messages
    if len(st.session_state.messages) == 0:
        chat_feedback.add_message("user", "Paracetamol là thuốc gì?")
        chat_feedback.add_message(
            "assistant",
            "Paracetamol là thuốc giảm đau, hạ sốt phổ biến. Liều dùng: 500-1000mg, 4-6 giờ/lần.",
            metadata={"agent": "drug_info", "confidence": 0.95}
        )
    
    chat_feedback.render_chat_with_feedback()
    
    # Statistics
    chat_feedback.render_sidebar_stats()
    
    # Show raw data
    with st.expander(" Raw Feedback Data"):
        st.json(chat_feedback.get_feedback_for_export())


if __name__ == "__main__":
    demo_feedback_ui()
