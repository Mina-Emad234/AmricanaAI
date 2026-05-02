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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import gradio as gr

# Load environment variables
load_dotenv()

# --- Configuration --- #
groq_api_key = os.environ.get("GROQ_API_KEY")
PDF_PATH = "/app/data/Privacy-Policy-Ar.pdf"
CHROMA_DIR = "/app/chroma_customer_support"

# --- Document Loading and Processing --- #
if not os.path.exists(PDF_PATH):
    print(f"Error: PDF file not found at {PDF_PATH}")
    exit(1)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
chunks = splitter.split_documents(docs)

# --- Vector Store Setup --- #
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it as an environment variable or in a .env file.")

print("Initializing embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Clear contents of Chroma directory safely (for Docker mounts)
if os.path.exists(CHROMA_DIR):
    print(f"Clearing old Chroma DB contents at {CHROMA_DIR}...")
    for item in os.listdir(CHROMA_DIR):
        item_path = os.path.join(CHROMA_DIR, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

print("Building Chroma DB from PDF chunks...")
os.makedirs(CHROMA_DIR, exist_ok=True)
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
print("Chroma DB built successfully.")

print("Initializing retriever...")
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
print("Retriever ready.")

# --- LLM Setup --- #
llm = ChatGroq(
    temperature=0.1, # Lower temperature for higher accuracy
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
)

# --- Prompts Engineering --- #

# 1. Contextualize Question Prompt
# This makes the AI smart enough to understand pronouns like "it", "they", or "him" based on history.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# 2. Main QA Prompt
system_prompt = (
    "You are a highly intelligent customer service assistant for Americana Restaurant."
    "\n\n"
    "GUIDELINES:"
    "1. Use the following pieces of retrieved context to answer the question."
    "2. If the answer is not in the context, politely state that you don't have this information "
    "and suggest contacting Americana customer support at 12345."
    "3. Language: Always respond in the SAME language used by the user (Arabic or English)."
    "4. Tone: Professional, friendly, and concise. Use emojis where appropriate."
    "\n\n"
    "CONTEXT:"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# --- Core Logic --- #

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def get_response(user_input, chat_history):
    # Step 1: Create a standalone question
    history_langchain = []
    for entry in chat_history:
        if isinstance(entry, dict):
            # Gradio 5.x / Message format
            role = entry.get("role")
            content = entry.get("content")
            if role == "user":
                history_langchain.append(HumanMessage(content=content))
            else:
                history_langchain.append(AIMessage(content=content))
        elif isinstance(entry, (list, tuple)):
            # Gradio 3.x/4.x / Tuple format
            if len(entry) == 2:
                human, ai = entry
                history_langchain.append(HumanMessage(content=human))
                history_langchain.append(AIMessage(content=ai))

    if history_langchain:
        standalone_question = contextualize_q_chain.invoke({
            "chat_history": history_langchain,
            "question": user_input
        })
    else:
        standalone_question = user_input

    # Step 2: Retrieve relevant documents
    docs = retriever.invoke(standalone_question)
    formatted_context = format_docs(docs)

    # Step 3: Generate final answer
    full_chain = qa_prompt | llm
    response = full_chain.invoke({
        "context": formatted_context,
        "question": standalone_question,
        "chat_history": history_langchain
    })
    
    return response.content

# --- Gradio Interface --- #

def chat_wrapper(message, history):
    bot_reply = get_response(message, history)
    yield bot_reply # Using yield for better integration with Gradio's chat flow

demo = gr.ChatInterface(
    fn=chat_wrapper,
    title="Americana Smart Support AI",
    description="Ask me anything about our policies and services!",
    examples=["What is the privacy policy?", "How do you handle my data?"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)