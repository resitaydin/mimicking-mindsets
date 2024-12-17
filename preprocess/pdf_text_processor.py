import os
import re
import pandas as pd
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
import regex

class PDFTextProcessor:
    def __init__(self, pdf_directory: str, output_directory: str, embedding_model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        """
        Initialize PDF preprocessor with input and output directories
        """
        self.pdf_directory = pdf_directory
        self.output_directory = output_directory
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            show_progress=True
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PDF using pdfplumber with proper encoding
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                text = " ".join(text_parts)
                
                # Get basic metadata
                metadata = {
                    "title": os.path.basename(pdf_path),
                    "creation_date": "Unknown",
                    "num_pages": len(pdf.pages)
                }
                
                return {
                    "text": text,
                    "title": metadata["title"],
                    "creation_date": metadata["creation_date"],
                    "num_pages": metadata["num_pages"]
                }
        except Exception as e:
            print(f"Text extraction failed for {pdf_path}: {e}")
            return {
                "text": "",
                "title": os.path.basename(pdf_path),
                "creation_date": "Unknown",
                "num_pages": 0
            }

    def clean_text(self, text: str) -> str:
        """
        Clean text while preserving case and Turkish characters
        """
        # Remove hyphenated line breaks
        text = text.replace('-\n', '').replace('-', '')
        
        # Remove special characters while preserving Turkish letters, numbers, and basic punctuation
        text = regex.sub(r'[^\p{L}\p{N}\s.,!?;:\-\'"\(\)]+', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # No need to convert to lowercase as we are using a case-sensitive model.
        
        return text
    
    def create_text_chunks(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        """
        Create text chunks using LangChain's RecursiveCharacterTextSplitter
        
        Parameters:
            text (str): The text to split into chunks
            chunk_size (int): Size of each chunk in characters (default: 500)
            chunk_overlap (int): Number of characters to overlap between chunks (default: 100)
        
        Returns:
            List[str]: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Ordered from most to least preferred split characters
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Line breaks
                ".",     # Sentences
                "!",     # Exclamations
                "?",     # Questions
                ";",     # Semi-colons (adding this)
                ":",     # Colons (adding this)
                ",",     # Commas
                " ",     # Words
                ""       # Characters
            ]
        )
        
        return text_splitter.split_text(text)
    
    def process_pdfs(self, chunk_size: int = 500, chunk_overlap: int = 100) -> pd.DataFrame:
        """
        Process all PDFs in the directory
        """
        processed_data = []
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        total_pdfs = len(pdf_files)
        
        print(f"Found {total_pdfs} PDF files to process")
        
        for idx, filename in enumerate(pdf_files, 1):
            print(f"\nProcessing PDF {idx}/{total_pdfs}: {filename}")
            pdf_path = os.path.join(self.pdf_directory, filename)
            
            # Extract text and metadata
            print("- Extracting text...")
            pdf_data = self.extract_text_from_pdf(pdf_path)
            
            # Clean text
            print("- Cleaning text...")
            cleaned_text = self.clean_text(pdf_data["text"])
            
            # Chunk text
            print("- Creating chunks...")
            text_chunks = self.create_text_chunks(cleaned_text, chunk_size, chunk_overlap)
            print(f"  Created {len(text_chunks)} chunks")
            
            # Create metadata for each chunk
            print("- Adding metadata...")
            for i, chunk in enumerate(text_chunks):
                processed_data.append({
                    'filename': filename,
                    'chunk_id': i,
                    'text': chunk,
                    'length': len(chunk),
                    'title': pdf_data["title"],
                    'creation_date': pdf_data["creation_date"],
                    'num_pages': pdf_data["num_pages"]
                })
        
        print("\nAll PDFs processed successfully!")
        print(f"Total chunks created: {len(processed_data)}")
        
        return pd.DataFrame(processed_data)
    
    def create_vector_store(self, df: pd.DataFrame) -> FAISS:
        """
        Create FAISS vector store from processed text chunks
        """
        texts = df['text'].tolist()
        
        # Modify metadata to include source field
        metadatas = []
        for _, row in df.iterrows():
            metadata = {
                'source': row['filename'],  # Add source field
                'chunk_id': row['chunk_id'],
                'title': row['title'],
                'creation_date': row['creation_date'],
                'num_pages': row['num_pages']
            }
            metadatas.append(metadata)
        
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        vector_store.save_local(os.path.join(self.output_directory, "vector_store"))
        return vector_store
    
    def save_processed_data(self, df: pd.DataFrame):
        """
        Save processed data to CSV and create vector store
        """
        print("\nSaving processed data...")
        
        # Save consolidated CSV
        csv_path = os.path.join(self.output_directory, 'processed_texts.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"- Saved CSV to: {csv_path}")
        
        # Create and save vector store
        print("- Creating vector store...")
        vector_store = self.create_vector_store(df)
        vector_store_path = os.path.join(self.output_directory, "vector_store")
        print(f"- Saved vector store to: {vector_store_path}")
        
        return vector_store

# Usage Example
if __name__ == "__main__":
    preprocessor = PDFTextProcessor(
        pdf_directory='works', 
        output_directory='output'
    )
    
    processed_df = preprocessor.process_pdfs(chunk_size=500, chunk_overlap=100)
    vector_store = preprocessor.save_processed_data(processed_df)
    
    print("PDF Processing Complete!")