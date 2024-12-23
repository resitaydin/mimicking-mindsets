# Mimicking Mindsets: AI-Generated Insights from Influential Minds

## Description
This project creates an AI-powered dialogue system that embodies the intellectual legacy of Prof. Dr. Erol Güngör, a prominent Turkish scholar. Using **Retrieval-Augmented Generation (RAG)**, the system provides responses that reflect Güngör's perspectives on social psychology, culture, and Turkish modernization. The AI assistant maintains the academic rigor and thoughtful analysis characteristic of Güngör's work while making his insights accessible through modern conversation.

## Key Features
- **Intelligent Response Generation**: Leverages Llama 3.3 70B model through Groq for generating contextually aware responses
- **Dynamic Confidence Scoring**: Real-time evaluation of response reliability with visual confidence metrics
- **Source Transparency**: Detailed citation system showing relevant source materials with similarity scores
- **Turkish Language Optimization**: Specialized for Turkish language processing using Turkish BERT embeddings
- **Interactive Web Interface**: Clean, modern UI built with Streamlit featuring:
  - Confidence gauges and metrics
  - Source material exploration
  - Response timing information
  - Chat history management

## Technical Stack
- **Frontend**: Streamlit with custom components and layouts
- **Language Model**: Llama 3.3 70B (via Groq)
- **Embeddings**: Turkish BERT (emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)
- **Vector Store**: ChromaDB for efficient similarity search
- **Document Processing**: PDFPlumber for text extraction
- **Framework**: LangChain for RAG pipeline orchestration

## System Components
- **RAGEngine**: Core component handling document retrieval and response generation
- **ResponseEvaluator**: Sophisticated confidence scoring system
- **PDFProcessor**: Intelligent document processing with semantic chunking, direct ChromaDB integration for vector storage

## Data Processing
- Semantic text chunking with configurable size and overlap
- Advanced PDF processing with metadata extraction
- Turkish-specific text cleaning and normalization
- Efficient vector storage and retrieval through ChromaDB

## Performance Features
- In-memory caching for improved response times
- Position-weighted similarity scoring
- Multi-stage confidence evaluation
- Configurable retrieval parameters for precision/recall balance

## Usage
The system provides an intuitive chat interface where users can:
1. Ask questions about Erol Güngör's work and thoughts
2. View confidence metrics for each response
3. Explore source materials and citations
4. Track response generation timing
5. Manage conversation history

## Future Development
- Additional Turkish scholars and their works
- Enhanced visualization options for knowledge relationships
- API endpoints for external integration
- Expanded document processing capabilities
- Multi-modal content support

## Requirements
- Python 3.8+
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Transformers
- PDFPlumber
- Groq Cloud API