from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class ErolGungorRAG:
    def __init__(self, vector_store_path: str = "output/vector_store"):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        self.vector_store = FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize language model
        self.llm = HuggingFaceHub(
            repo_id="ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1", 
            model_kwargs={
                "temperature": 0.3,  # Lower temperature for more focused responses
                "max_new_tokens": 512,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        )
        
        # Create the QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create a simple QA chain"""
        template = """Aşağıdaki bağlam bilgisini kullanarak, Prof. Dr. Erol Güngör'ün bakış açısıyla soruyu yanıtla.
        Yanıtı detaylı ve açıklayıcı bir şekilde ver, gereksiz giriş cümleleri kullanma.
        Eğer bağlam bilgisinde soruyla ilgili yeterli bilgi yoksa, sadece
        "Bu konu hakkında eserlerimde detaylı bir bilgi bulunmamaktadır." yanıtını ver. kaynakça bölümü yazma.
        
        Bağlam:
        {context}
        
        Soru: {question}
        
        Erol Güngör'ün yanıtı:"""
        
        PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
            validate_template=True
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "fetch_k": 5
                }
            ),
            chain_type_kwargs={
                "prompt": PROMPT,
                "verbose": False
            },
            return_source_documents=True
        )
    
    def get_response(self, query: str) -> dict:
        """Get response with sources"""
        result = self.qa_chain({"query": query})
        
        # Clean up the response by removing the prompt and context
        response_text = result["result"]
        
        # Remove the prompt template if it appears in the response
        if "Bağlam:" in response_text:
            response_text = response_text.split("Bağlam:")[-1]
        if "Soru:" in response_text:
            response_text = response_text.split("Soru:")[-1]
        if "Erol Güngör'ün yanıtı:" in response_text:
            response_text = response_text.split("Erol Güngör'ün yanıtı:")[-1]
        
        # Clean up any leading/trailing whitespace
        response_text = response_text.strip()
        
        sources = []
        total_relevance = 0
        
        # Calculate confidence based on source relevance
        for doc in result["source_documents"]:
            # You might want to implement a more sophisticated relevance calculation
            relevance = self.vector_store.similarity_search_with_score(query, k=1)[0][1]
            total_relevance += relevance
            
            sources.append({
                "file": doc.metadata["filename"],
                "text": doc.page_content[:200] + "..."
            })
        
        # Calculate average confidence score (normalized between 0 and 1)
        avg_confidence = min(1.0, total_relevance / len(result["source_documents"]))
        
        # Reduce confidence if the response indicates no information
        if "detaylı bir bilgi bulunmamaktadır" in response_text.lower():
            avg_confidence *= 0.5
        
        return {
            "response": response_text,
            "sources": sources,
            "confidence_score": round(avg_confidence, 2)  # Round to 2 decimal places
        }