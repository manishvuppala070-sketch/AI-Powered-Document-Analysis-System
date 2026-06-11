# AI-Powered Intelligent Document Analysis System
> Upload any PDF and ask questions from it using AI-powered semantic retrieval and automated passage highlighting.

 **Live Demo:** [ai-document-analysis-manish.streamlit.app](https://ai-document-analysis-manish.streamlit.app)

---

## Features

- **PDF Question Answering** — Ask natural language questions directly from any uploaded PDF
- **Semantic Retrieval** — Uses FAISS vector search to find the most relevant document passages
- **Automated Highlighting** — Highlights the source passage in the original PDF for traceability
- **Streamlit Interface** — Clean, interactive UI accessible without any coding knowledge
- **Offline Processing** — Document processing runs locally with no data sent to external servers

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Frontend | Streamlit |
| Vector Search | FAISS |
| PDF Processing | PyMuPDF, pdfplumber |
| Embeddings | Sentence Transformers |
| QA Model | deepset/minilm-uncased-squad2 |
| NLP | Transformers, NLTK |
| Version Control | Git, GitHub |

---

## Project Structure

```
Intelligent_Document_Analysis/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore
└── utils/
    └── rag_pipeline.py     # RAG pipeline: retrieval + QA logic
```

---

## How It Works

1. Upload a PDF document through the Streamlit interface
2. The system extracts and chunks the text, then builds a FAISS vector index
3. Ask any question in natural language
4. The system retrieves the most relevant passages using semantic similarity
5. A QA model extracts the precise answer from the retrieved context
6. Source page and passage are displayed alongside the answer

---

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/manishvuppala070-sketch/AI-Powered-Document-Analysis-System.git
cd AI-Powered-Document-Analysis-System

# Install dependencies
pip install -r requirements.txt

# Run the app
py -3.11 -m streamlit run app.py
```

---

## Future Improvements

- Multi-document support for cross-document querying
- Improved PDF highlighting accuracy
- Advanced summarization using larger language models
- Export answers with highlighted PDF as a report

---

## Author

**Vuppala Manish**
- GitHub: [manishvuppala070-sketch](https://github.com/manishvuppala070-sketch)
- LinkedIn: [manish-vuppala-01390a311](https://www.linkedin.com/in/manish-vuppala-01390a311/)
- Email: manishvuppala070@gmail.com