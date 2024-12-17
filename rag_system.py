from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
import langchain
from typing import Dict, Any

# Enable caching
langchain.cache = InMemoryCache()

class ErolGungorRAG:
    def __init__(self, vector_store_path: str = "output/vector_store"):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        try:
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            raise ValueError(f"Failed to load vector store: {str(e)}")
        
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
        """Create an optimized QA chain with better prompting"""
        prompt_template = """Rol: Sen Prof. Dr. Erol Güngör'ün düşüncelerini ve eserlerini temsil eden bir yapay zeka asistanısın.

        Bağlam: {context}

        Soru: {question}

        Yanıt verirken şu kurallara uy:
        1. Sadece verilen bağlam içindeki bilgileri kullan
        2. Emin olmadığın konularda spekülasyon yapma
        3. Erol Güngör'ün akademik ve düşünce tarzına uygun bir dil kullan
        4. Yanıtı mümkün olduğunca açık ve anlaşılır şekilde ver

        Yanıt:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3},
                search_type="mmr"
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response with error handling and confidence scoring"""
        try:
            result = self.qa_chain({"query": question})
            
            # Extract sources with improved metadata handling
            sources = []
            for doc in result.get("source_documents", []):
                # Get source from metadata
                source_file = doc.metadata.get("source", "Unknown")
                
                # Clean up source filename if needed
                if isinstance(source_file, str):
                    source_file = source_file.split("/")[-1].split("\\")[-1]
                
                sources.append({
                    "file": source_file,
                    "text": doc.page_content
                })
            
            # Calculate confidence score based on source relevance
            confidence_score = min(len(sources) * 0.33, 1.0)
            
            return {
                "response": result["result"],
                "confidence_score": confidence_score,
                "sources": sources
            }
        except Exception as e:
            return {
                "response": "Üzgünüm, bu soruyu yanıtlarken bir hata oluştu.",
                "confidence_score": 0.0,
                "sources": [],
                "error": str(e)
            }