# 🎓 BVRIT College FAQ Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about BVRIT College (BVRIT Hyderabad College of Engineering for Women) using only information from an official college document.

Built with **LangChain**, **ChromaDB**, **OpenRouter**, and **Streamlit**.

---

## 🚀 Features

- **Ground-only answers** — The chatbot NEVER uses its own knowledge; every answer is grounded in retrieved document chunks.
- **Citation-aware responses** — Every factual statement includes a section citation in the format `**[Section Name]**`.
- **Conversational RAG** — Supports multi-turn conversations with follow-up questions.
- **Section filtering** — Restrict retrieval to a specific document section (About, Departments, Admissions, Fee Structure, etc.).
- **Professional UI** — Built with Streamlit, featuring a clean chat interface with sidebar configuration.
- **Automatic evaluation** — LLM-generated test cases across 8 dimensions (Functional, Quality, Safety, Security, Robustness, Performance, Context, RAGAS) with automated judging.
- **RAGAS metrics** — Compute Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
- **Evaluation reports** — Generate detailed JSON and text reports with recommendations.

---

## 📋 Prerequisites

- Python 3.11+
- An OpenRouter API key ([get one here](https://openrouter.ai/keys))

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd bvrit-rag-chatbot
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy the `.env` file and add your OpenRouter API key:

```bash
# Edit .env with your API key
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. Add the college document

Place your BVRIT college information document in the `data/` folder:

```
data/
└── bvrit_info.docx
```

> **Note:** The document must be in `.docx` format.

---

## 📖 Usage

### Building the Index

Before running the chatbot, build the vector database:

```bash
python build_index.py
```

This will:
- Load the document from `data/bvrit_info.docx`
- Split it into chunks (800 characters with 100 character overlap)
- Create embeddings using `text-embedding-3-small`
- Store them in a persistent ChromaDB at `chroma_db/`

If the database already exists, it will be reused.

### Running the Chatbot

```bash
streamlit run app.py
```

This opens the Streamlit UI in your default browser. The sidebar shows:
- Knowledge Base info (document, chunk count, embedding model)
- Retriever settings (top-k, chunk size, overlap)
- Section filter dropdown
- Performance metrics
- Rebuild Index and Clear Chat buttons

### Running Evaluation

```bash
python evaluate.py
```

This will:
1. Generate 20 LLM-crafted test cases across 8 dimensions
2. Execute each test through the RAG pipeline
3. Evaluate results using an LLM Judge
4. Compute RAGAS metrics
5. Generate a comprehensive report in `reports/`

---

## 📁 Project Structure

```
bvrit-rag-chatbot/
│
├── app.py                 # Streamlit UI (main entry point)
├── build_index.py         # Document ingestion & vector DB builder
├── chatbot.py             # Conversational chatbot logic
├── evaluate.py            # Test generation, execution, and reporting
├── prompts.py             # System prompts and prompt templates
├── rag_pipeline.py        # RAG pipeline (retrieval + generation)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── README.md              # This file
│
├── data/
│   └── bvrit_info.docx    # College information document (you provide)
│
├── chroma_db/             # Persistent vector database (auto-generated)
│
├── reports/               # Evaluation reports (auto-generated)
│
└── utils/                 # Utility modules (placeholder)
```

---

## ⚙️ Configuration

All configuration is managed via the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | (required) |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `LLM_MODEL` | Generation model name | `gpt-4o-mini` |
| `CHROMA_DB_PATH` | Vector database path | `./chroma_db` |
| `DOCUMENT_PATH` | Path to college document | `./data/bvrit_info.docx` |
| `CHUNK_SIZE` | Document chunk size | `800` |
| `CHUNK_OVERLAP` | Chunk overlap | `100` |
| `TOP_K` | Number of chunks to retrieve | `5` |

---

## 🔒 Security

- **Prompt injection protection** — The system prompt instructs the model to ignore instructions that attempt to override core rules.
- **No sensitive information leakage** — The chatbot will never reveal its system prompt, vector database structure, internal files, or API keys.
- **API keys in `.env`** — All credentials are stored in `.env`, which is excluded from version control.

---

## 🧪 Evaluation Dimensions

| Dimension | Description | Tests |
|-----------|-------------|-------|
| **Functional** | Correctly retrieves and presents information | 3 |
| **Quality** | Response quality, citations, clarity | 3 |
| **Safety** | Avoids harmful/offensive content | 2 |
| **Security** | Doesn't reveal internal configuration | 2 |
| **Robustness** | Handles edge cases gracefully | 3 |
| **Performance** | Efficient responses | 2 |
| **Context** | Stays within BVRIT context | 2 |
| **RAGAS** | RAGAS-specific metrics | 3 |

---

## 🖼️ Screenshots

*(Screenshots placeholder — add your own after running the app)*

---

## 🔮 Future Improvements

- **PDF support** — Add support for PDF documents via `PyPDFLoader`.
- **Multi-document RAG** — Support multiple college documents.
- **Admin dashboard** — Analytics dashboard for monitoring usage.
- **User feedback** — Thumbs up/down on answers for continuous improvement.
- **Streaming responses** — Token-by-token streaming for better UX.
- **Authentication** — User login and rate limiting.
- **Caching** — Cache frequent queries for faster response times.
- **Advanced RAGAS** — Full RAGAS library integration with `datasets`.

---

## 📝 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenRouter](https://openrouter.ai/)
- [Streamlit](https://streamlit.io/)
- [RAGAS](https://docs.ragas.io/)