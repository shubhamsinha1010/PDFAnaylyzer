# Document Query and Chat Engine

This project uses LlamaIndex, HuggingFace, and Groq LLM to index documents (`.pdf`, `.docx`) and answer queries or engage in chat with context memory.

## Features

- Loads and indexes documents from the current directory
- Uses Groq Llama3-70B LLM for answering questions
- Embeds documents with BAAI/bge-small-en-v1.5
- Supports both direct queries and contextual chat with memory

## Requirements

- Python 3.8+
- [llama-index](https://pypi.org/project/llama-index/)
- [huggingface_hub](https://pypi.org/project/huggingface-hub/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
