# DocuMind AI

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple)

A document question-answering app built using RAG (Retrieval Augmented Generation). Upload any text document and ask questions about it - the AI reads the document and gives accurate answers.

## What it does

- Upload a `.txt` file
- The app splits the document into small chunks
- Chunks are converted to embeddings and stored in ChromaDB
- When you ask a question, it finds the most relevant chunks
- Groq AI (LLaMA model) generates an answer using those chunks
- Shows source chunks so you can verify the answer

## Tech Stack

- **Python** - Backend language
- **Streamlit** - Web interface
- **LangChain** - For text splitting, embeddings, and LLM integration
- **Groq AI** - LLM provider (uses LLaMA 3.3 70B model)
- **ChromaDB** - Vector database for storing and searching text
- **HuggingFace Embeddings** - For converting text to vectors (all-MiniLM-L6-v2)

## How to run

1. Clone the repo
```bash
git clone https://github.com/pragatisingh1556/documind-ai.git
cd documind-ai
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Add your Groq API key in `.env` file
```
GROQ_API_KEY=your_api_key_here
```

4. Run the app
```bash
streamlit run app.py
```

5. Open the URL shown in terminal (usually http://localhost:8501)

## How to get Groq API key

1. Go to https://console.groq.com
2. Sign up for free
3. Go to API Keys section
4. Create a new key and paste it in `.env` file

## Project Structure

```
documind-ai/
  app.py              # main streamlit app
  requirements.txt    # python dependencies
  sample.txt          # sample document for testing
  .env                # groq api key (not pushed to github)
```

## Screenshots

_Coming soon_

## How RAG works (simple explanation)

1. **Split** - Document is split into small pieces (chunks)
2. **Embed** - Each chunk is converted to a number vector using HuggingFace model
3. **Store** - Vectors are stored in ChromaDB (vector database)
4. **Search** - When you ask a question, it finds similar chunks using vector similarity
5. **Answer** - The relevant chunks + your question are sent to Groq AI to generate answer

This way the AI only answers from your document, not from its general knowledge.

## Author

**Pragati Singh** - 2025 Graduate
