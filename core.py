import langchain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.cache import InMemoryCache
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from typing import Dict, Any, List
import time
import numpy as np

# Enable caching
langchain.cache = InMemoryCache()

class RAGEngine:
    def __init__(
        self, 
        chroma_path: str = "data/chromadb",
        collection_name: str = "erol_gungor_docs"
    ):
        """Initialize RAG system with ChromaDB"""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB and create vector store
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            self.collection = self.chroma_client.get_collection(collection_name)
            
            # Create Langchain vector store wrapper
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize ChromaDB: {str(e)}")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            temperature=0.2,
            model_name="llama-3.3-70b-versatile",
            max_tokens=2048,
            streaming=True
        )
        
        # Initialize confidence scorer
        self.confidence_scorer = ResponseEvaluator()
        
        # Create the QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create an optimized QA chain with improved retrieval"""
        prompt_template = """Rol: Sen Prof. Dr. Erol Güngör'ün düşüncelerini ve eserlerini temsil eden bir yapay zeka asistanısın.

        Bağlam: {context}

        Soru: {question}

        Yanıt verirken şu kurallara uy:
        1. Sadece verilen bağlam içindeki bilgileri kullan
        2. Eğer bağlamda soruyla ilgili hiçbir bilgi yoksa, bunu açıkça belirt ve spekülasyon yapma
        3. Erol Güngör'ün akademik ve düşünce tarzına uygun bir dil kullan
        4. Yanıtı mümkün olduğunca açık ve anlaşılır şekilde ver
        5. Erol Güngör'müş gibi davran ve onun adına konuş
        6. Eğer soru Erol Güngör'ün çalışma alanı dışındaysa, bunu nazikçe belirt

        Yanıt:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "fetch_k": 10,
                    "lambda_mult": 0.8
                },
                search_type="mmr",
                score_threshold=0.85
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def _process_sources(self, source_documents: List) -> List[Dict]:
        """Process source documents into a standardized format"""
        sources = []
        for doc in source_documents:
            metadata = doc.metadata
            sources.append({
                "file": metadata.get("source_file", "Unknown"),
                "title": metadata.get("title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "chunk_id": metadata.get("chunk_id", 0),
                "text": doc.page_content,
                "similarity": metadata.get("similarity", 0.0)
            })
        return sources
    
    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response with enhanced error handling and source tracking"""
        try:
            # Get initial response
            result = self.qa_chain({"query": question})
            
            # Process sources
            sources = self._process_sources(result.get("source_documents", []))
            
            # Calculate comprehensive confidence score
            confidence_score = self.confidence_scorer.calculate_confidence(
                query=question,
                source_documents=sources
            )
            
            # Adjust response based on confidence
            if confidence_score < 0.3:
                result["result"] = "Bu soru, Prof. Dr. Erol Güngör'ün çalışma alanı dışında kalmaktadır."
            elif confidence_score < 0.5:
                result["result"] += "\n\nNot: Bu yanıt sınırlı kaynaklara dayanmaktadır."
            
            return {
                "response": result["result"],
                "confidence_score": confidence_score,
                "sources": sources,
                "metadata": {
                    "total_chunks": self.collection.count(),
                    "query_time": time.time(),
                    "source_count": len(sources)
                }
            }
            
        except Exception as e:
            return {
                "response": "Üzgünüm, bu soruyu yanıtlarken bir hata oluştu.",
                "confidence_score": 0.0,
                "sources": [],
                "error": str(e),
                "metadata": {
                    "error_type": type(e).__name__
                }
            }
        
class ResponseEvaluator:
    def __init__(
        self,
        min_similarity_threshold: float = 0.3,
        medium_similarity_threshold: float = 0.4,
        weight_decay_factor: float = 0.8
    ):
        """
        Initialize the confidence scorer with adjusted thresholds
        
        Args:
            min_similarity_threshold: Minimum similarity score to consider relevant
            medium_similarity_threshold: Threshold for medium confidence
            weight_decay_factor: Factor to decay weights for subsequent documents
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.medium_similarity_threshold = medium_similarity_threshold
        self.weight_decay_factor = weight_decay_factor
        
        # Initialize the same embeddings model as the RAG system
        self.embeddings = HuggingFaceEmbeddings(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def calculate_confidence(
        self,
        query: str,
        source_documents: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score with adjusted thresholds
        """
        if not source_documents:
            return 0.0
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate weighted similarity scores
            weighted_scores = []
            for idx, doc in enumerate(source_documents):
                doc_text = doc.get('text', '')
                if not doc_text:
                    continue
                    
                doc_embedding = self.embeddings.embed_query(doc_text)
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                doc['similarity'] = float(similarity)
                
                # Apply position-based weight decay
                weight = self.weight_decay_factor ** idx
                weighted_scores.append(similarity * weight)
            
            if not weighted_scores:
                return 0.0
                
            avg_similarity = np.mean(weighted_scores)
            print(f"Debug - Average similarity: {avg_similarity}")
            
            if avg_similarity < self.min_similarity_threshold:
                confidence = 0.0
            elif avg_similarity < self.medium_similarity_threshold:
                # Scale between 0 and 0.5 for low confidence
                confidence = 0.5 * (avg_similarity - self.min_similarity_threshold) / (self.medium_similarity_threshold - self.min_similarity_threshold)
            else:
                # Scale between 0.5 and 1.0 for high confidence
                confidence = 0.5 + 0.5 * (avg_similarity - self.medium_similarity_threshold) / (1 - self.medium_similarity_threshold)
            
            print(f"Debug - Final confidence score: {confidence}")
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            print(f"Error calculating confidence score: {str(e)}")
            return 0.0
            
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)