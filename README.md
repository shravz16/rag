
# RAGProcessor

RAGProcessor is a Python-based system for Retrieval-Augmented Generation (RAG), enabling intelligent question answering over document collections. It integrates AWS SQS, S3, Pinecone, and OpenAI to process documents (PDF, DOCX), generate embeddings, and answer queries using GPT-4.
Features

    Document Processing: Extracts text from PDFs and DOCX files.
    Embedding: Converts document chunks into vector embeddings using OpenAI's text-embedding-ada-002.
    Vector Search: Stores embeddings in Pinecone for efficient retrieval.
    Query Answering: Answers queries using GPT-4 based on document content.

Setup

    Install dependencies:

pip install boto3 langchain langchain_openai pinecone-client langchain-pinecone PyPDF2 python-docx

    Configure AWS credentials:

aws configure

    Set environment variables for Pinecone and OpenAI API keys:

export PINECONE_API_KEY="your-pinecone-api-key"
export OPENAI_API_KEY="your-openai-api-key"

Usage

Initialize and process messages:

processor = RAGProcessor(
    queue_url="your-sqs-queue-url",
    pinecone_api_key="your-pinecone-api-key",
    openai_api_key="your-openai-api-key"
)

processor.process_messages()

License

This project is licensed under the MIT License.
