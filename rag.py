from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from docx import Document as DocxDocument
import time
import PyPDF2
os.environ["PINECONE_API_KEY"] = "1f784203-27de-487e-90be-b2bff7b3b8f8"

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "docs-rag-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    print(text)
    return text
def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        # Open the PDF file in binary mode
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)

            # Extract text from each page
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract text from the page
                text += page.extract_text() + "\n"

        print("PDF text extracted successfully!")
        return text

    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None
# Input: Path to your .docx file
docx_file_path = "Shravan kumar resume.docx"
pdf_file_path = "Shravan Resume dec 2024 - Google Docs.pdf"
document_text = extract_text_from_pdf(pdf_file_path)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
doc_chunks = text_splitter.split_text(document_text)

# Convert text chunks to Document objects
documents = [
    Document(
        page_content=chunk,
        metadata={"source": docx_file_path}
    ) for chunk in doc_chunks
]

# Initialize LangChain embedding object
model_name = "multilingual-e5-large"
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# Embed and upsert the chunks into Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=documents,  # Now using Document objects
    index_name="docs-rag-chatbot",
    embedding=embeddings,
    namespace="plain-docx-namespace"
)

# Allow time for processing
time.sleep(1)
print("Embedding and upsert of plain .docx content completed successfully!")

index = pc.Index(index_name)
namespace = "plain-docx-namespace"

for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0],
        namespace=namespace,
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    print(query)



retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever=docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
print(retrieval_chain)
query1 = "does shravan know Generative AI?"

answer1_with_knowledge = retrieval_chain.invoke({"input": query1})


print("Query 1:", query1)
print("Answer with knowledge:\n", str(answer1_with_knowledge['answer']))
print()


