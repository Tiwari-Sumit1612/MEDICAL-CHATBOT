from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split
)

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found")

print("Pinecone:", bool(PINECONE_API_KEY))

# =====================================================
# Load PDF
# =====================================================

extracted_data = load_pdf_file(data="data/")

filtered_data = filter_to_minimal_docs(extracted_data)

text_chunks = text_split(filtered_data)

print(f"Total Chunks: {len(text_chunks)}")

# =====================================================
# Pinecone Hosted Embeddings
# =====================================================

embeddings = PineconeEmbeddings(
    model="multilingual-e5-large"
)

# =====================================================
# Pinecone Connection
# =====================================================

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot-v2"

# =====================================================
# Create Index
# =====================================================

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

print(f"Using index: {index_name}")

# =====================================================
# Upload in Batches
# =====================================================

batch_size = 100

total_batches = (len(text_chunks) + batch_size - 1) // batch_size

for i in range(0, len(text_chunks), batch_size):

    batch = text_chunks[i:i + batch_size]

    PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )

    print(
        f"Uploaded Batch "
        f"{i // batch_size + 1}/{total_batches}"
    )

print("Indexing completed successfully!")