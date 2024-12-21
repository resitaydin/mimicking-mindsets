import os
import re
import chromadb
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
import regex
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

class PDFTextProcessor:
    def __init__(
        self, 
        pdf_directory: str, 
        output_directory: str, 
        embedding_model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        collection_name: str = "erol_gungor_docs"
    ):
        """
        Initialize PDF preprocessor with ChromaDB integration using consistent embedding model
        """
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize embeddings with the Turkish BERT model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(output_directory, "chromadb"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
        )
        
        # Create or get collection with the Turkish model embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "dimension": 768
            },
            embedding_function=sentence_transformer_ef
        )   
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text and metadata from PDF using pdfplumber
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                text = " ".join(text_parts)
                
                # Extract more detailed metadata
                metadata = {
                    "title": os.path.basename(pdf_path),
                    "creation_date": pdf.metadata.get('CreationDate', 'Unknown'),
                    "author": pdf.metadata.get('Author', 'Unknown'),
                    "num_pages": len(pdf.pages),
                    "producer": pdf.metadata.get('Producer', 'Unknown')
                }
                
                return {**metadata, "text": text}
        except Exception as e:
            print(f"Text extraction failed for {pdf_path}: {e}")
            return {
                "text": "",
                "title": os.path.basename(pdf_path),
                "creation_date": "Unknown",
                "author": "Unknown",
                "num_pages": 0,
                "producer": "Unknown"
            }

    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving Turkish characters and essential formatting
        """
        # Remove hyphenated line breaks
        text = text.replace('-\n', '').replace('-', '')
        
        # Remove special characters while preserving Turkish letters
        text = regex.sub(r'[^\p{L}\p{N}\s.,!?;:\-\'"\(\)]+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_text_chunks(
        self, 
        text: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100
    ) -> List[str]:
        """
        Create semantic text chunks using improved splitting strategy
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Line breaks
                ".",     # Sentences
                "!",     # Exclamations
                "?",     # Questions
                ";",     # Semi-colons
                ":",     # Colons
                ",",     # Commas
                " ",     # Words
                ""       # Characters
            ]
        )
        
        return text_splitter.split_text(text)
    
    def process_pdfs(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100
    ) -> None:
        """
        Process PDFs and store in ChromaDB with improved metadata
        """
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        total_pdfs = len(pdf_files)
        
        print(f"Found {total_pdfs} PDF files to process")
        
        for idx, filename in enumerate(pdf_files, 1):
            print(f"\nProcessing PDF {idx}/{total_pdfs}: {filename}")
            pdf_path = os.path.join(self.pdf_directory, filename)
            
            # Extract text and metadata
            print("- Extracting text and metadata...")
            pdf_data = self.extract_text_from_pdf(pdf_path)
            
            # Clean text
            print("- Cleaning text...")
            cleaned_text = self.clean_text(pdf_data.pop("text"))
            
            # Create chunks
            print("- Creating chunks...")
            text_chunks = self.create_text_chunks(cleaned_text, chunk_size, chunk_overlap)
            print(f"  Created {len(text_chunks)} chunks")
            
            # Prepare documents for ChromaDB
            ids = [f"{filename}_{i}" for i in range(len(text_chunks))]
            metadatas = [{
                **pdf_data,
                "chunk_id": i,
                "source_file": filename,
            } for i in range(len(text_chunks))]
            
            # Add to ChromaDB
            print("- Adding to ChromaDB...")
            self.collection.add(
                documents=text_chunks,
                ids=ids,
                metadatas=metadatas
            )
        
        print("\nAll PDFs processed and stored in ChromaDB!")
        print(f"Total chunks stored: {self.collection.count()}")

# Usage Example
if __name__ == "__main__":
    preprocessor = PDFTextProcessor(
        pdf_directory='erol_gungor_docs', 
        output_directory='output'
    )
    preprocessor.process_pdfs(chunk_size=500, chunk_overlap=100)