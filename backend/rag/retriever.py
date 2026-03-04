"""
Document retrieval for RAG system - Truy xuất tài liệu cho hệ thống RAG
"""
from typing import List, Dict, Any, Optional
from backend.rag.embeddings import get_embedding_service
from backend.rag.vector_store import get_default_vector_store
from backend.config import settings
from backend.utils import app_logger, chunk_text
from backend.models import DrugQuery
import time


class Retriever:
    """Retriever tài liệu cho RAG"""
    
    def __init__(
        self,
        vector_store=None,
        embedding_service=None,
        top_k: int = None,
        similarity_threshold: float = None
    ):
        """
        Khởi tạo retriever
        
        Args:
            vector_store: Instance vector store
            embedding_service: Instance embedding service
            top_k: Số lượng documents cần truy xuất
            similarity_threshold: Ngưỡng điểm tương đồng tối thiểu
        """
        self.vector_store = vector_store or get_default_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.top_k = top_k or settings.TOP_K_RESULTS
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        app_logger.info(
            f"Initialized retriever with top_k={self.top_k}, "
            f"threshold={self.similarity_threshold}"
        )
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất documents liên quan cho một query
        
        Args:
            query: Truy vấn tìm kiếm
            top_k: Số lượng kết quả trả về (ghi đè mặc định)
            filter_dict: Bộ lọc metadata tùy chọn
            
        Returns:
            Danh sách documents liên quan với metadata
        """
        start_time = time.time()
        
        try:
            # Tạo query embedding
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Tìm kiếm vector store
            k = top_k or self.top_k
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k * 2,  # Lấy nhiều kết quả hơn để lọc
                filter_dict=filter_dict
            )
            
            # Lọc theo ngưỡng tương đồng
            filtered_results = [
                result for result in results
                if result['similarity'] >= self.similarity_threshold
            ][:k]
            
            duration = time.time() - start_time
            app_logger.info(
                f"Retrieved {len(filtered_results)} documents in {duration:.3f}s "
                f"for query: {query[:100]}"
            )
            
            return filtered_results
            
        except Exception as e:
            app_logger.error(f"Lỗi khi truy xuất documents: {e}")
            raise
    
    async def retrieve_drug_info(
        self,
        drug_name: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất thông tin thuốc
        
        Args:
            drug_name: Tên thuốc
            top_k: Số lượng kết quả
            
        Returns:
            Danh sách documents thông tin thuốc
        """
        filter_dict = {"type": "drug_info"}
        return await self.retrieve(
            query=f"thuốc {drug_name} thông tin chi tiết",
            top_k=top_k,
            filter_dict=filter_dict
        )
    
    async def retrieve_interactions(
        self,
        drug_names: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất thông tin tương tác thuốc
        
        Args:
            drug_names: Danh sách tên thuốc
            top_k: Số lượng kết quả
            
        Returns:
            Danh sách documents tương tác
        """
        query = f"tương tác giữa {' và '.join(drug_names)}"
        filter_dict = {"type": "interaction"}
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
    
    async def retrieve_dosage_info(
        self,
        drug_name: str,
        patient_info: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất thông tin liều lượng
        
        Args:
            drug_name: Tên thuốc
            patient_info: Thông tin bệnh nhân (tuổi, cân nặng, v.v.)
            top_k: Số lượng kết quả
            
        Returns:
            Danh sách documents liều lượng
        """
        query_parts = [f"liều lượng thuốc {drug_name}"]
        
        if patient_info:
            if "age" in patient_info:
                query_parts.append(f"tuổi {patient_info['age']}")
            if "weight" in patient_info:
                query_parts.append(f"cân nặng {patient_info['weight']}kg")
        
        query = " ".join(query_parts)
        filter_dict = {"type": "dosage"}
        
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
    
    async def retrieve_side_effects(
        self,
        drug_name: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Truy xuất thông tin tác dụng phụ
        
        Args:
            drug_name: Tên thuốc
            top_k: Số lượng kết quả
            
        Returns:
            Danh sách documents tác dụng phụ
        """
        query = f"tác dụng phụ của thuốc {drug_name}"
        filter_dict = {"type": "side_effects"}
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
    
    async def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm lai (semantic + keyword)
        
        Args:
            query: Truy vấn tìm kiếm
            keywords: Keywords bổ sung để lọc
            top_k: Số lượng kết quả
            
        Returns:
            Danh sách documents
        """
        # Hiện tại, chỉ sử dụng tìm kiếm semantic
        # TODO: Triển khai keyword boosting
        results = await self.retrieve(query, top_k)
        
        # Nếu có keywords, tăng điểm các kết quả chứa keywords
        if keywords:
            for result in results:
                keyword_count = sum(
                    1 for keyword in keywords
                    if keyword.lower() in result['document'].lower()
                )
                result['keyword_boost'] = keyword_count
            
            # Sắp xếp lại theo điểm kết hợp
            results.sort(
                key=lambda x: x['similarity'] + (x.get('keyword_boost', 0) * 0.1),
                reverse=True
            )
        
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Định dạng documents đã truy xuất thành chuỗi context
        
        Args:
            results: Danh sách documents đã truy xuất
            
        Returns:
            Chuỗi context đã định dạng
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            context_parts.append(
                f"[Nguồn {i}: {source}]\n{result['document']}\n"
            )
        
        return "\n---\n".join(context_parts)


# Global retriever instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Lấy hoặc tạo instance retriever toàn cục"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
