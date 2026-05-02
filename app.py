import os
import shutil
import glob
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import gradio as gr

# Load environment variables from .env file
load_dotenv()

# --- Configuration --- #

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it as an environment variable or in a .env file.")

PDF_PATH = "/app/data/Privacy-Policy-Ar.pdf"
CHROMA_DIR = "/app/chroma_customer_support"

# --- Document Loading and Splitting --- #

if not os.path.exists(PDF_PATH):
    print(f"Error: PDF file not found at {PDF_PATH}")
    exit(1)

print(f"Loading PDF from: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split PDF into {len(chunks)} chunks.")

# --- Embeddings --- #

print("Initializing embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# --- Vector Store (ChromaDB) --- #
# Always wipe and rebuild so the AI always uses the latest PDF

# Clear contents inside the mounted volume directory (cannot delete the root mount itself)
if os.path.exists(CHROMA_DIR):
    print(f"Clearing old Chroma DB contents at {CHROMA_DIR}...")
    # Remove sqlite file
    for f in glob.glob(os.path.join(CHROMA_DIR, "*.sqlite3")):
        os.remove(f)
    # Remove data subdirectories
    for item in os.listdir(CHROMA_DIR):
        item_path = os.path.join(CHROMA_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print("Old Chroma DB cleared.")

os.makedirs(CHROMA_DIR, exist_ok=True)

print("Building Chroma DB from PDF...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
print(f"Chroma DB built with {vectordb._collection.count()} documents.")

# Set up the retriever
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
)
print("Retriever initialized.")

# --- Language Model (LLM) Setup --- #

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

print("Initializing ChatGroq LLM...")
llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
)

# System Prompt
system_prompt = """
You are a restaurant customer service assistant.
Use only the information provided in the context below.
If the information is missing, state that you do not know and suggest contacting human support.
Language Policy: Automatically detect and match the user's language (e.g., Arabic, English, or any other language used).
Keep responses short, clear, and professional.
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)
print("RAG chain assembled successfully.")

# --- Gradio Interface --- #

def respond(message, chat_history):
    if not message or not message.strip():
        return chat_history, ""

    try:
        response = rag_chain.invoke(message)
        bot_reply = response.content
    except Exception as e:
        bot_reply = f"Sorry, an error occurred: {str(e)}"

    if chat_history is None:
        chat_history = []

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_reply})

    return chat_history, ""

print("Launching Gradio interface...")
with gr.Blocks() as demo:
    gr.Markdown("## Amrecana Restaurant Smart Customer Service (RAG Bot)")

    chatbot = gr.Chatbot(label="Customer Service AI Assistant")

    msg = gr.Textbox(
        placeholder="اكتب استفسارك هنا...",
        label="رسالتك"
    )

    clear = gr.Button("مسح الدردشة")

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)