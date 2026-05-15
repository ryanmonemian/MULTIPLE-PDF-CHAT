# Document Q&A Chatbot (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) application that enables users to "chat" with their PDF documents. This tool utilizes vector similarity search to retrieve relevant context from uploaded files and feeds it into a Large Language Model (LLM) to generate accurate, source-grounded answers with minimal hallucinations.

## Features

* **RAG Architecture:** Ingests unstructured PDF data and allows for natural language querying.
* **Vector Search:** Uses **FAISS** for low-latency, high-performance similarity search across document embeddings.
* **Grounded Responses:** Engineered system prompts to restrict the LLM to answer *only* using the provided document context, ensuring accuracy.
* **Multi-Turn Memory:** Maintains conversation history for a seamless chat experience.
* **Interactive UI:** Built with **Streamlit** to support drag-and-drop file uploads and real-time processing indicators.

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** [LangChain](https://python.langchain.com/) (Chains, Prompts, Memory)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **LLM & Embeddings:** OpenAI API (`gpt-4o`, `text-embedding-ada-002`)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Processing:** PyPDF2

## How It Works

1.  **Ingestion:** The app reads uploaded PDF files and extracts raw text.
2.  **Chunking:** Text is split into smaller, semantic chunks (e.g., 1000 characters) to fit within the context window.
3.  **Embedding:** Chunks are converted into vector embeddings using OpenAI's embedding model.
4.  **Storage:** Vectors are stored locally in a FAISS index for fast retrieval.
5.  **Retrieval & Generation:** When a user asks a question, the system finds the top 3-4 most similar text chunks and sends them to the LLM with the prompt: *"Answer the user's question using only this context."*

## Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/rag-chatbot.git](https://github.com/yourusername/rag-chatbot.git)
    cd rag-chatbot
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=sk-your-api-key-here
    ```

5.  **Run the application**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Open the local URL provided by Streamlit (usually `http://localhost:8501`).
2.  Upload one or multiple PDF files using the sidebar.
3.  Click **"Process"** to build the vector index.
4.  Type your question in the chat input to query your documents.

## Dashboard Preview
<img width="2858" height="1208" alt="1" src="https://github.com/user-attachments/assets/ab7f9f98-3d2a-4911-bcc1-fc77a30fa49e" />




## Future Improvements

* Add support for other file formats (DOCX, TXT).
* Implement "Source Highlighting" to show exactly which page the answer came from.
* Experiment with open-source local embeddings (HuggingFace) to reduce API costs.
