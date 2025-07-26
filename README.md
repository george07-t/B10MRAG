# B10MRAG: Bangla 10 Minute RAG System ![MIT License](https://img.shields.io/badge/license-MIT-green.svg) ![AI](https://img.shields.io/badge/Domain-AI-purple.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

A production-grade Retrieval-Augmented Generation (RAG) system for answering both Bangla and English queries from a PDF knowledge base. Features include session-based chat, robust memory, and a FastAPI interface.

---

## 🚀 Setup Guide

### 1. System Dependencies

Before installing Python requirements, install **Tesseract** and Bengali language support:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ben
```

### 2. Python Environment

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Requirements

```bash
pip install -r requirements.txt
```

### 4. Start the API Server

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --workers 4 --reload
```

---

## 🛠️ Used Tools, Libraries, and Packages

- **LangChain**: Core RAG, chunking, and vector search
- **LangGraph**: Agentic workflow and session management
- **Ollama**: Local LLM and embedding model serving
- **ChromaDB**: Vector store for semantic retrieval
- **FastAPI**: Production API server
- **multilingual-pdf2text**: Robust PDF text extraction with OCR (Bangla/English)
- **sentence-transformers**: For reranking and advanced embedding (optional)
- **Tesseract**: OCR engine for Bangla/English PDF extraction

---

## 💬 Sample Queries and Outputs

### Bangla

**Query:**  
`অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`  
**Output:**  
`উত্তর: শুম্ভু নাথ`

**Query:**  
`কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`  
**Output:**  
`উত্তর: মামা`

### English

**Query:**  
`What is the real age of Kalyani at the time of marriage?`  
**Output:**  
`Answer: ১৫ বছর (15 years)`

---

## 📖 API Documentation

### `/status` (GET)
- **Description:** Returns system readiness.
- **Response:**  
    `{ "status": "ready" }` or `{ "status": "not_ready" }`

### `/chat` (POST)
- **Description:** Submit a query and get an answer.
- **Body:**  
    ```json
    {
        "query": "Your question here",
        "session_id": "optional-session-id"
    }
    ```
- **Response:**  
    ```json
    {
        "answer": "The answer",
        "session_id": "session-id"
    }
    ```

### `/sessions` (GET)
- **Description:** List all session IDs.

### `/session/{session_id}` (GET)
- **Description:** Get chat history for a session.

### `/session/{session_id}` (DELETE)
- **Description:** Delete a single session.

### `/sessions` (DELETE)
- **Description:** Delete all sessions.

---

## 📊 Evaluation Matrix

See `rag_evaluation.py` for automated evaluation.

- **Accuracy:** Percentage of test queries where the expected answer is found in the generated answer.
- Sample test cases are provided in the script.

---

## ❓ Q&A: Design Choices and Rationale

**Q: What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**  
A:  
We use `multilingual-pdf2text` with Tesseract OCR and Bengali language support. This library is robust for Bangla/English mixed PDFs and handles scanned images. Formatting challenges included inconsistent line breaks and OCR noise, which we mitigate with Unicode normalization and regex-based cleaning.

**Q: What chunking strategy did you choose? Why do you think it works well for semantic retrieval?**  
A:  
We use a sentence/paragraph-aware chunking strategy with `RecursiveCharacterTextSplitter`, chunk size 700, overlap 120, and Bangla/English sentence separators (।, ?, !, \n). This ensures each chunk is semantically meaningful and not cut mid-sentence, improving retrieval accuracy.

**Q: What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**  
A:  
We use `all-minilm:latest` via Ollama for embeddings. This model is lightweight, multilingual, and captures semantic similarity well for both Bangla and English, making it ideal for our hybrid corpus.

**Q: How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**  
**A:**  
We use a hybrid retrieval strategy:

- Vector similarity search (cosine similarity) in ChromaDB, powered by MiniLM embeddings, to capture semantic similarity between queries and document chunks.
- BM25 keyword search for exact and partial keyword matches, which helps with out-of-vocabulary or rare terms.

Combining both methods improves retrieval robustness for Bangla/English and ensures both semantic and lexical matches. ChromaDB is chosen for its speed and integration with LangChain, while BM25 adds classic IR strength for keyword-heavy queries.

**Q: How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**  
A:  
We use semantic embeddings and chunking that preserves context. If a query is vague, the retriever may return less relevant chunks, and the LLM will either answer with the closest match or respond with "উত্তর পাওয়া যায়নি।" ("Not found.") if no relevant context is found.

**Q: Do the results seem relevant? If not, what might improve them?**  
A:  
Results are generally relevant for well-formed queries. Improvements could include:
- Finer-tuned chunking (e.g., dynamic chunk size)
- Using a larger or more Bangla-specialized embedding model
- Adding BM25 or hybrid retrieval
- Expanding the knowledge base

---

## 📝 Example: Evaluation Script

See `rag_evaluation.py` for automated test cases and accuracy calculation.

---

## 🧑‍💻 Contributors

- George Tonmoy Roy

---

## License
This project is licensed under the MIT License. See the [MIT LICENSE](./LICENSE) file for details.

For any issues or contributions, please open an issue or pull request on GitHub.
