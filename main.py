from dotenv import load_dotenv
import os
import streamlit as st
from huggingface_hub import login
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import tempfile

load_dotenv()

if os.path.exists(".env"):
    load_dotenv(".env")  # For local development
else:
    # For Streamlit Cloud
    from streamlit.runtime.secrets import secrets
    os.environ["HUGGING_FACE_KEY"] = secrets["HUGGING_FACE_KEY"]
    os.environ["GROQ_API_KEY"] = secrets["GROQ_API_KEY"]

# Initialize the system
def initialize_system():
    # Set tokenizers parallelism environment variable to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    login(token=os.environ["HUGGING_FACE_KEY"])

    # Initialize LLM
    llm = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

    # Initialize Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    return llm, embed_model


# Process uploaded PDF and create index
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        documents = SimpleDirectoryReader(input_files=[tmp_file_path]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        os.unlink(tmp_file_path)  # Delete the temporary file
        return index
    except Exception as e:
        os.unlink(tmp_file_path)  # Clean up even if error occurs
        st.error(f"Error processing PDF: {e}")
        return None


# Main Streamlit app
def main():
    st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

    st.title("📄 PDF Q&A Assistant")
    st.markdown("Upload a PDF and ask questions about its content")

    # Initialize session state
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file:
            with st.spinner("Processing PDF..."):
                st.session_state.index = process_pdf(uploaded_file)
                if st.session_state.index:
                    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
                        st.session_state.index.as_retriever(),
                        memory=ChatMemoryBuffer.from_defaults(token_limit=3900),
                        llm=st.session_state.llm
                    )
                    st.success("PDF processed successfully!")

    # Initialize system components
    if 'llm' not in st.session_state:
        with st.spinner("Initializing AI components..."):
            st.session_state.llm, _ = initialize_system()

    # Main chat interface
    if st.session_state.index is None:
        st.info("Please upload a PDF file to get started")
    else:
        st.subheader("Ask a question about the PDF")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chat_engine.chat(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()