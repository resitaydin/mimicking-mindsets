from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain_community.vectorstores import Chroma
import langchain
import chromadb
from typing import Dict, Any, List
import numpy as np

# Enable caching
langchain.cache = InMemoryCache()

class ErolGungorRAG:
    def __init__(
        self, 
        chroma_path: str = "output/chromadb",
        collection_name: str = "erol_gungor_docs"
    ):
        """
        Initialize RAG system with ChromaDB
        """
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
        
        # Create the QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create an optimized QA chain with improved retrieval"""
        prompt_template = """Rol: Sen Prof. Dr. Erol Güngör'ün düşüncelerini ve eserlerini temsil eden bir yapay zeka asistanısın.

        Bağlam: {context}

        Soru: {question}

        Yanıt verirken şu kurallara uy:
        1. Sadece verilen bağlam içindeki bilgileri kullan
        2. Emin olmadığın konularda spekülasyon yapma
        3. Erol Güngör'ün akademik ve düşünce tarzına uygun bir dil kullan
        4. Yanıtı mümkün olduğunca açık ve anlaşılır şekilde ver
        5. Erol Güngör'müş gibi davran ve onun adına konuş

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
                    "k": 5,
                    "fetch_k": 20  # Fetch more documents initially for MMR
                },
                search_type="mmr",  # Use Maximum Marginal Relevance
                score_threshold=0.7  # Minimum similarity score threshold
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def calculate_confidence_score(self, sources: List[Dict]) -> float:
        """
        Calculate a more sophisticated confidence score based on multiple factors
        """
        if not sources:
            return 0.0
        
        # Factors to consider:
        # 1. Number of relevant sources (normalized)
        source_count_score = min(len(sources) / 5, 1.0)
        
        # 2. Average semantic similarity (if available)
        similarity_scores = [s.get('similarity', 0.7) for s in sources]
        avg_similarity = np.mean(similarity_scores)
        
        # 3. Source diversity (unique documents)
        unique_sources = len(set(s['file'] for s in sources))
        diversity_score = unique_sources / len(sources)
        
        # Combine scores with weights
        final_score = (
            0.4 * source_count_score +
            0.4 * avg_similarity +
            0.2 * diversity_score
        )
        
        return round(final_score, 2)
    
    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response with enhanced error handling and source tracking"""
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Process source documents
            sources = []
            for doc in result.get("source_documents", []):
                metadata = doc.metadata
                sources.append({
                    "file": metadata.get("source_file", "Unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "text": doc.page_content,
                    "similarity": metadata.get("similarity", 0.7)
                })
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(sources)
            
            return {
                "response": result["result"],
                "confidence_score": confidence_score,
                "sources": sources,
                "metadata": {
                    "total_chunks": self.collection.count(),
                    "query_time": result.get("query_time", None)
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
# Usage Example
if __name__ == "__main__":
    rag = ErolGungorRAG()
    response = rag.get_response("Erol Güngör'ün kültür anlayışı nedir?")
    print(f"Confidence: {response['confidence_score']}")
    print(f"Response: {response['response']}")
    print("\nSources:")
    for source in response['sources']:
        print(f"- {source['file']}: {source['title']}")