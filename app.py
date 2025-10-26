import streamlit as st
from typing import List, Dict
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app
st.title("Chatbot with Document Retrieval")
st.write("Upload a PDF document and chat with it!")

# Initialize session states
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False

def validate_api_key(api_key):
    """Validate OpenAI API key."""
    if not api_key or not api_key.startswith('sk-'):
        return False
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",  # Use cheaper model for validation
            max_tokens=10
        )
        # Try a simple completion to validate the key
        llm.invoke("Hi")
        return True
    except Exception as e:
        st.error(f"API Key Error: {str(e)}")
        return False

# Get API keys
if not st.session_state.api_key_valid:
    # Try to get from environment/secrets first
    env_key = os.getenv("OPENAI_API_KEY") or (
        st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    )
    
    if env_key and validate_api_key(env_key):
        st.session_state.openai_api_key = env_key
        st.session_state.api_key_valid = True
    else:
        # Show API key input if no valid key is found
        api_key_input = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            key="api_key_input"
        )
        if api_key_input:
            if validate_api_key(api_key_input):
                st.session_state.openai_api_key = api_key_input
                st.session_state.api_key_valid = True
                st.success("API key is valid! You can now use the chatbot.")
                st.rerun()

# Get HuggingFace token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or (
    st.secrets.get("HUGGINGFACEHUB_API_TOKEN") if hasattr(st, "secrets") else None
)
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Initialize embeddings with explicit CPU configuration
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Add a button to reset API key if needed
if st.session_state.api_key_valid:
    if st.sidebar.button("Reset API Key"):
        st.session_state.api_key_valid = False
        st.session_state.openai_api_key = None
        st.rerun()

if st.session_state.api_key_valid:
    try:
        llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model="gpt-4",  # Using GPT-4 model
            temperature=0.7,
            max_tokens=4096
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI: {str(e)}")
        st.session_state.api_key_valid = False
        st.rerun()

    # Initialize session state for messages if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        accept_multiple_files=True
    )

    # Initialize retriever in session state if not present
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    if uploaded_file:
        documents = []
        for pdf_file in uploaded_file:
            # Create a temporary file with a unique name based on the uploaded file's name
            temp_pdf_path = f"./temp_{pdf_file.name}"
            
            # Save the uploaded file content
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and process the PDF
            try:
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                documents.extend(docs)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("Documents processed and ready for querying!")

    # Create the template for question-answering
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and within three sentences.

    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer: """

    qa_prompt = ChatPromptTemplate.from_template(template)

    # Helper function for chat history management
    def get_chat_history() -> List[Dict]:
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        return st.session_state.messages

    # Create a chain for processing questions with context
    def process_query(query: str):
        # Get chat history
        messages = get_chat_history()
        
        # Format chat history for display
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        if st.session_state.retriever is None:
            response_content = "Please upload a PDF document first before asking questions."
        else:
            # Get relevant documents using invoke
            docs = st.session_state.retriever.invoke(query)
            context = "\n".join(doc.page_content for doc in docs)
            
            # Generate response using the LLM
            response = llm.invoke(f"""Context: {context}\n\nChat History: {history_text}\n\nQuestion: {query}\n\nAnswer:""")
            response_content = response.content
        
        # Update chat history
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": response_content})
        
        return response_content

    if st.session_state.retriever is None:
        st.warning("Please upload a PDF document to start asking questions.")
    
    user_input = st.text_input("Your question:", disabled=(st.session_state.retriever is None))
    if user_input:
        with st.spinner('Processing your question...'):
            response = process_query(user_input)
            st.success(f"Assistant: {response}")
        
        # Display chat history
        st.write("Chat History:")
        for message in get_chat_history():
            role = message["role"]
            content = message["content"]
            if role == "user":
                st.info(f"User: {content}")
            elif role == "assistant":
                st.success(f"Assistant: {content}")

else:
    st.warning("Please provide your OpenAI API Key to continue.")





