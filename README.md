# 📚 Smart Research Assistant

A RAG-based AI assistant that allows users to ask natural-language questions about research papers, technical documents, text files, Word documents, and web content.

The system retrieves relevant information from the provided sources and uses an LLM to generate answers based on the retrieved context, along with supporting sources and retrieved chunks.

---

## 🚀 Features

- 📄 Upload PDF documents
- 📝 Upload TXT files
- 📝 Upload DOCX files
- 🌐 Retrieve content from public web pages
- 📚 Support multiple documents and sources simultaneously
- ✂️ Recursive text chunking
- 🔢 Hugging Face embeddings
- 🗄️ ChromaDB vector database
- 🔍 Semantic similarity search
- 🤖 LLM-based answer generation
- 📑 Source and page references
- 🔎 View retrieved document chunks
- 📊 RAG evaluation using Ragas
- 🎨 Streamlit-based user interface
- ⚠️ Handles documents with no readable text gracefully

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
             PDF / TXT / DOCX / Web
                       │
                       ▼
                Text Extraction
                       │
                       ▼
                  Text Chunking
                       │
                       ▼
             Hugging Face Embeddings
                       │
                       ▼
                   ChromaDB
                       │
                       ▼
              Semantic Retrieval
                       │
                   Top-K Chunks
                       │
                       ▼
                     LLM
                       │
                       ▼
              Generated Answer
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Sources       Retrieved Chunks
1. Document Ingestion

The application accepts multiple types of sources:

PDF files
TXT files
DOCX files
Public webpage URLs

The appropriate extraction method is selected based on the source type.

2. Text Chunking

Extracted content is divided into smaller chunks using RecursiveCharacterTextSplitter.

Current configuration:

Chunk size: 500 characters
Chunk overlap: 50 characters

Chunking allows the retrieval system to work with smaller and more relevant pieces of information.

3. Embeddings

Each text chunk is converted into a numerical vector using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings capture the semantic meaning of the document chunks.

4. Vector Storage

The generated embeddings are stored in ChromaDB.

Each source is associated with its own vector collection, allowing multiple documents to be processed and searched.

5. Retrieval

When the user asks a question, semantic similarity search is performed across the available document sources.

The system currently retrieves:

Top-K = 3

relevant chunks.

6. Answer Generation

The retrieved chunks are provided to the language model as context.

The LLM then generates an answer based on the retrieved information.

7. Source Attribution

The application displays the source documents and page information associated with the retrieved chunks.

Users can also expand the retrieved-context section to inspect the chunks used for generating the answer.

📊 RAG Evaluation

The RAG pipeline was evaluated using Ragas on an evaluation test set.

Metric	Score
Faithfulness	1.00
Answer Relevancy	0.86
Context Precision	0.90
Context Recall	0.80
Evaluation Metrics

Faithfulness
Measures whether the generated answer is supported by the retrieved context.

Answer Relevancy
Measures how relevant the generated answer is to the user's question.

Context Precision
Measures whether the retrieved chunks are relevant to the question.

Context Recall
Measures whether the retrieved context contains the information required to answer the question.

The displayed scores are based on the project's Ragas evaluation test set.

🛠️ Tech Stack
Category	Technologies
Programming	Python
RAG / AI	LangChain, Hugging Face Sentence Transformers
Vector Database	ChromaDB
Evaluation	Ragas
Document Processing	PyPDF, python-docx
Web Extraction	BeautifulSoup, Requests
Application	Streamlit
Development	Git, GitHub, VS Code
📁 Project Structure
Smart-Research-Assistant/
│
├── app.py
├── rag_utils.py
├── generate.py
├── evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
app.py

Handles the Streamlit application and user interaction:

Document upload
Web URL input
Session state
Question input
Answer display
Source display
Retrieved chunks
Ragas evaluation display
rag_utils.py

Contains the document processing and retrieval functionality:

PDF text extraction
TXT text extraction
DOCX text extraction
Webpage text extraction
Recursive text chunking
Embedding creation
ChromaDB vector store creation
Multi-document retrieval
generate.py

Handles:

LLM initialization
Prompt construction
Context-based answer generation
evaluation.py

Contains the Ragas evaluation workflow used to evaluate the RAG pipeline.

⚙️ Installation
1. Clone the repository
git clone https://github.com/ishaanagnihotri18/Smart-Research-Assistant.git
cd Smart-Research-Assistant
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

Windows:

venv\Scripts\activate

macOS / Linux:

source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
🔐 Environment Variables

Create a .env file in the project root.

Add the API key required by the LLM configured in generate.py.

Example:

GEMINI_API_KEY=your_api_key_here

Never commit .env or API keys to GitHub.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

💬 Example Questions

After uploading a document, you can ask questions such as:

What are the main findings of this paper?
What methodology was used?
What are the key conclusions?
What are the limitations mentioned in the document?
What evidence supports this conclusion?
Which document discusses this topic?

When multiple sources are uploaded, questions can also require information from different documents.

🔍 Example RAG Flow

For a question such as:

What are the main findings of the research paper?

the system performs:

User Question
      │
      ▼
Semantic Search
      │
      ▼
Top 3 Relevant Chunks
      │
      ▼
LLM + Retrieved Context
      │
      ▼
Generated Answer
      │
      ▼
Supporting Sources
⚠️ Limitations
The PDF pipeline relies on extractable text. Scanned or image-only PDFs may not contain readable text.
Web extraction works best with publicly accessible webpages containing server-rendered text.
JavaScript-heavy or login-protected websites may not be extractable.
The current retrieval configuration uses a fixed Top-K value of 3.
Ragas scores depend on the evaluation dataset and may vary with different documents and questions.
🔮 Future Improvements

Possible future improvements include:

OCR support for scanned PDFs
Improved web-page extraction
Configurable chunk size and overlap
Configurable Top-K retrieval
Conversation history
Improved citation formatting
Larger-scale document management
Expanded evaluation datasets
🎯 Project Objective

The objective of this project is to build a practical RAG-based research assistant that can retrieve relevant information from user-provided sources and generate answers based on the retrieved context.

The project demonstrates the complete RAG workflow:

Document Ingestion
        ↓
Text Chunking
        ↓
Embeddings
        ↓
Vector Storage
        ↓
Semantic Retrieval
        ↓
LLM Generation
        ↓
Source Attribution
        ↓
RAG Evaluation
👨‍💻 Author

Ishaan Agnihotri

B.Tech CSE — Artificial Intelligence & Machine Learning

GitHub# 📚 Smart Research Assistant

A RAG-based AI assistant that allows users to ask natural-language questions about research papers, technical documents, text files, Word documents, and web content.

The system retrieves relevant information from the provided sources and uses an LLM to generate answers based on the retrieved context, along with supporting sources and retrieved chunks.

---

## 🚀 Features

- 📄 Upload PDF documents
- 📝 Upload TXT files
- 📝 Upload DOCX files
- 🌐 Retrieve content from public web pages
- 📚 Support multiple documents and sources simultaneously
- ✂️ Recursive text chunking
- 🔢 Hugging Face embeddings
- 🗄️ ChromaDB vector database
- 🔍 Semantic similarity search
- 🤖 LLM-based answer generation
- 📑 Source and page references
- 🔎 View retrieved document chunks
- 📊 RAG evaluation using Ragas
- 🎨 Streamlit-based user interface
- ⚠️ Handles documents with no readable text gracefully

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
             PDF / TXT / DOCX / Web
                       │
                       ▼
                Text Extraction
                       │
                       ▼
                  Text Chunking
                       │
                       ▼
             Hugging Face Embeddings
                       │
                       ▼
                   ChromaDB
                       │
                       ▼
              Semantic Retrieval
                       │
                   Top-K Chunks
                       │
                       ▼
                     LLM
                       │
                       ▼
              Generated Answer
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Sources       Retrieved Chunks
1. Document Ingestion

The application accepts multiple types of sources:

PDF files
TXT files
DOCX files
Public webpage URLs

The appropriate extraction method is selected based on the source type.

2. Text Chunking

Extracted content is divided into smaller chunks using RecursiveCharacterTextSplitter.

Current configuration:

Chunk size: 500 characters
Chunk overlap: 50 characters

Chunking allows the retrieval system to work with smaller and more relevant pieces of information.

3. Embeddings

Each text chunk is converted into a numerical vector using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings capture the semantic meaning of the document chunks.

4. Vector Storage

The generated embeddings are stored in ChromaDB.

Each source is associated with its own vector collection, allowing multiple documents to be processed and searched.

5. Retrieval

When the user asks a question, semantic similarity search is performed across the available document sources.

The system currently retrieves:

Top-K = 3

relevant chunks.

6. Answer Generation

The retrieved chunks are provided to the language model as context.

The LLM then generates an answer based on the retrieved information.

7. Source Attribution

The application displays the source documents and page information associated with the retrieved chunks.

Users can also expand the retrieved-context section to inspect the chunks used for generating the answer.

📊 RAG Evaluation

The RAG pipeline was evaluated using Ragas on an evaluation test set.

Metric	Score
Faithfulness	1.00
Answer Relevancy	0.86
Context Precision	0.90
Context Recall	0.80
Evaluation Metrics

Faithfulness
Measures whether the generated answer is supported by the retrieved context.

Answer Relevancy
Measures how relevant the generated answer is to the user's question.

Context Precision
Measures whether the retrieved chunks are relevant to the question.

Context Recall
Measures whether the retrieved context contains the information required to answer the question.

The displayed scores are based on the project's Ragas evaluation test set.

🛠️ Tech Stack
Category	Technologies
Programming	Python
RAG / AI	LangChain, Hugging Face Sentence Transformers
Vector Database	ChromaDB
Evaluation	Ragas
Document Processing	PyPDF, python-docx
Web Extraction	BeautifulSoup, Requests
Application	Streamlit
Development	Git, GitHub, VS Code
📁 Project Structure
Smart-Research-Assistant/
│
├── app.py
├── rag_utils.py
├── generate.py
├── evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
app.py

Handles the Streamlit application and user interaction:

Document upload
Web URL input
Session state
Question input
Answer display
Source display
Retrieved chunks
Ragas evaluation display
rag_utils.py

Contains the document processing and retrieval functionality:

PDF text extraction
TXT text extraction
DOCX text extraction
Webpage text extraction
Recursive text chunking
Embedding creation
ChromaDB vector store creation
Multi-document retrieval
generate.py

Handles:

LLM initialization
Prompt construction
Context-based answer generation
evaluation.py

Contains the Ragas evaluation workflow used to evaluate the RAG pipeline.

⚙️ Installation
1. Clone the repository
git clone https://github.com/ishaanagnihotri18/Smart-Research-Assistant.git
cd Smart-Research-Assistant
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

Windows:

venv\Scripts\activate

macOS / Linux:

source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
🔐 Environment Variables

Create a .env file in the project root.

Add the API key required by the LLM configured in generate.py.

Example:

GEMINI_API_KEY=your_api_key_here

Never commit .env or API keys to GitHub.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

💬 Example Questions

After uploading a document, you can ask questions such as:

What are the main findings of this paper?
What methodology was used?
What are the key conclusions?
What are the limitations mentioned in the document?
What evidence supports this conclusion?
Which document discusses this topic?

When multiple sources are uploaded, questions can also require information from different documents.

🔍 Example RAG Flow

For a question such as:

What are the main findings of the research paper?

the system performs:

User Question
      │
      ▼
Semantic Search
      │
      ▼
Top 3 Relevant Chunks
      │
      ▼
LLM + Retrieved Context
      │
      ▼
Generated Answer
      │
      ▼
Supporting Sources
⚠️ Limitations
The PDF pipeline relies on extractable text. Scanned or image-only PDFs may not contain readable text.
Web extraction works best with publicly accessible webpages containing server-rendered text.
JavaScript-heavy or login-protected websites may not be extractable.
The current retrieval configuration uses a fixed Top-K value of 3.
Ragas scores depend on the evaluation dataset and may vary with different documents and questions.
🔮 Future Improvements

Possible future improvements include:

OCR support for scanned PDFs
Improved web-page extraction
Configurable chunk size and overlap
Configurable Top-K retrieval
Conversation history
Improved citation formatting
Larger-scale document management
Expanded evaluation datasets
🎯 Project Objective

The objective of this project is to build a practical RAG-based research assistant that can retrieve relevant information from user-provided sources and generate answers based on the retrieved context.

The project demonstrates the complete RAG workflow:

Document Ingestion
        ↓
Text Chunking
        ↓
Embeddings
        ↓
Vector Storage
        ↓
Semantic Retrieval
        ↓
LLM Generation
        ↓
Source Attribution
        ↓
RAG Evaluation
👨‍💻 Author

Ishaan Agnihotri

B.Tech CSE — Artificial Intelligence & Machine Learning

GitHub