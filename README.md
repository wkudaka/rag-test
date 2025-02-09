# Document Retrieval with LangChain and ChromaDB

This repository contains an example pipeline for loading documents in .docx format, indexing them in a vector database (ChromaDB), and querying them using a large language model (LLM) via LangChain.

## Technologies Used
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Docx2txt](https://pypi.org/project/docx2txt/)
- [Ollama (LLama3)](https://ollama.com/)

## Project Structure
```
/
├── rag_with_docx.py      # Main script
├── requirements.txt      # Project dependencies
├── README.md             # Repository documentation
```

## Installation and Setup

### 1. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Script
```sh
python rag_with_docx.py
```

## How It Works
1. **Loads Documents:** The script retrieves `.docx` files from the `documents/` folder.
2. **Splits into Chunks:** Uses `CharacterTextSplitter` to divide documents into smaller parts.
3. **Generates Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to convert text into vectors.
4. **Stores in ChromaDB:** Saves embeddings in the `vector_db/` vector database.
5. **Performs Queries:** Uses an LLM to answer questions based on the documents.

## Customization
- To change the embedding model, modify `EMBEDDING_MODEL_NAME` in `rag_with_docx.py`.
- To use a different LLM, update `MODEL_NAME` and `LLAMA_ENDPOINT`.
