from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import numpy as np

"""
This script tests the vector store with similarity search and detailed analysis.
"""

def test_vector_store_with_scores():
    print("Loading embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    try:
        vector_store = FAISS.load_local(
            "output/vector_store", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load vector store: {str(e)}")
        return
    
    # Comprehensive test queries
    test_queries = [
        "öğrenme süreçleri",
        "eğitim yöntemleri",
        "araştırma metodolojisi",
        "sonuç ve tartışma"
    ]
    
    print("\n📊 Detailed Vector Store Retrieval Analysis:")
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            # Use similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=3)
            
            # Detailed analysis of each result
            for i, (result, score) in enumerate(results, 1):
                print(f"\n[Result {i}]")
                print(f"Similarity Score: {score:.4f}")
                print(f"Source: {result.metadata['filename']}")
                print(f"Chunk ID: {result.metadata['chunk_id']}")
                print(f"Text Preview: {result.page_content[:300]}...")
                
                # Additional context analysis
                print(f"Text Length: {len(result.page_content)} characters")
        
            # Calculate retrieval statistics
            scores = [score for _, score in results]
            print("\n📈 Retrieval Statistics:")
            print(f"Average Similarity Score: {np.mean(scores):.4f}")
            print(f"Lowest Similarity Score: {np.min(scores):.4f}")
            print(f"Highest Similarity Score: {np.max(scores):.4f}")
        
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")

def analyze_vector_store_performance():
    print("\n🔬 Comprehensive Vector Store Performance Analysis")
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load vector store
    vector_store = FAISS.load_local(
        "output/vector_store", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Load processed data for comprehensive analysis
    df = pd.read_csv("output/processed_texts.csv", encoding='utf-8')
    
    # Performance metrics
    print(f"Total Chunks: {len(df)}")
    print(f"Unique Sources: {df['filename'].nunique()}")

if __name__ == "__main__":
    test_vector_store_with_scores()
    analyze_vector_store_performance()