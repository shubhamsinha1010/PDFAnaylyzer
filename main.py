from dotenv import load_dotenv
load_dotenv()
import os

from huggingface_hub import login
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# 1. Set tokenizers parallelism environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
login(token=os.environ["HUGGING_FACE_KEY"])

# 2. Initialize LLM - Make sure GROQ_API_KEY is set in your environment
llm = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

# 3. Initialize Embedding Model - Using a more reliable public model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 4. Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512  # Optimal chunk size for most cases

# 5. Load documents
try:
    de_tools_blog = SimpleDirectoryReader(
        "./", 
        required_exts=[".pdf", ".docx"]
    ).load_data()
except Exception as e:
    print(f"Error loading documents: {e}")
    exit(1)

# 6. Create index
try:
    index = VectorStoreIndex.from_documents(de_tools_blog)
except Exception as e:
    print(f"Error creating index: {e}")
    exit(1)

# 7. Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# 8. First query
try:
    response = query_engine.query("What is a rate limiter")
    print(response)
except Exception as e:
    print(f"Query failed: {e}")

# 9. Setup chat engine with memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm
)

try:
    response = query_engine.query("What is leaky bucket algorithm")
    print(response)
except Exception as e:
    print(f"Query failed: {e}")
