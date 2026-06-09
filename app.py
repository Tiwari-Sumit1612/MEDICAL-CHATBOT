from flask import Flask, render_template, request

from src.prompt import *
from langchain_pinecone import PineconeEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os

# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# =====================================================
# Flask App
# =====================================================

app = Flask(__name__)

# =====================================================
# Embeddings
# =====================================================

embeddings = PineconeEmbeddings(
    model="multilingual-e5-large"
)

# =====================================================
# Pinecone Vector Store
# =====================================================

index_name = "medical-chatbot-v2"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# =====================================================
# Groq LLM
# =====================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# =====================================================
# Prompt
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# =====================================================
# RAG Chain
# =====================================================

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# =====================================================
# Routes
# =====================================================

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    print(f"\nQuestion: {msg}")

    response = rag_chain.invoke(
        {
            "input": msg
        }
    )

    answer = response["answer"]

    print(f"\nAnswer: {answer}")

    return str(answer)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )