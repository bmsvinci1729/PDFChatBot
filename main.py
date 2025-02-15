import os
# The os module in Python provides operating system functionalities,
# such as handling environment variables, file paths, and system interactions.
import streamlit as st
from dotenv import load_dotenv # to load theenv variables from .env file
import pdfplumber # to extract text from pdfs
from langchain.text_splitter import RecursiveCharacterTextSplitter # chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # embedding models and gpt-4
from langchain_community.vectorstores import FAISS # vector database which focus on faster storage and retrieval than the memory (stores bothe embeddings and the text
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# API handling
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("⚠️ OPENAI_API_KEY not found! Set it in .env file or as an environment variable.")

#################################################################################################################

# pdf extraction
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# chunking with overlaps (nearly doubled the total length)
def chunk_text(text, chunk_size=750, chunk_overlap=150):
    """Splits extracted text into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


# pdf_path = "/home/bms/Sumedha/rag-chatbot/statement-of-purpose-1.pdf"
# raw_text = extract_text_from_pdf(pdf_path)
# chunks = chunk_text(raw_text)


# embedding and vectorbase storage
def create_vector_store(chunks):
    """Converts chunks into embeddings and stores them in FAISS."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(chunks, embedding_model)
    return vector_store

# vector_store = create_vector_store(chunks)
# vector_store.save_local("faiss_index")


# retrieval based on the query using similarity search (cosine similarity)
def retrieve_top_k(query, vector_store, k=7):  
    """Retrieves the most relevant chunks for the given query."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)  # 🔥 Corrected retrieval method


# GPT 4 model
def generate_response(query, retrieved_texts, custom_prompt, conversation_history):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.8, max_tokens=2048, openai_api_key=openai_api_key)

    context = "\n\n".join(retrieved_texts)
    history_context = "\n".join([f"User: {msg['query']}\nBot: {msg['response']}" for msg in conversation_history])
    full_prompt = f"""
    {custom_prompt}

    Conversation History:
    {history_context}

    Use the following context to provide a detailed and comprehensive answer:
    
    Context:
    {context}
    
    Question: {query}

    Provide a well-structured response and include all relevant details.
    """

    response = llm.invoke([SystemMessage(content=full_prompt)])  
    return response.content


# STREAMLIT HOSTING CODES

st.set_page_config(page_title=" RAG-Based PDF Chatbot", layout="wide")
st.title("SUMEDHA Chatbot - Ask Anything from Your Documents!")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [] 

uploaded_file = st.file_uploader(" Upload a PDF", type="pdf")

if uploaded_file:
    st.info("Processing PDF...")

    raw_text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(raw_text)

    # Creating vector database
    vector_store = create_vector_store(chunks)
    vector_store.save_local("faiss_index")

    st.success(" PDF processed! Now, India wants to know answers to your questions.")

    query = st.text_input(" Ask a question from the PDF:")

    # Custom Prompt Input
    custom_prompt = st.text_area(
        " Customize the AI Prompt (Optional):",
        value="You are an AI assistant answering questions based on the given document. Use the provided context to respond."
    )

    if query:
        with st.spinner(" Searching for answers..."):
            retrieved_docs = retrieve_top_k(query, vector_store)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            response = generate_response(query, retrieved_texts, custom_prompt, st.session_state.conversation_history)

            st.session_state.conversation_history.append({"query": query, "response": response})

        st.markdown("###  Chatbot Conversation:")
        for chat in st.session_state.conversation_history:
            st.markdown(f"** User:** {chat['query']}")
            st.markdown(f"** Bot:** {chat['response']}")
            st.markdown("---") 