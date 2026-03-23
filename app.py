# documind ai - document question answering app
# upload a txt file and ask questions about it
# uses groq ai for answering and chromadb for searching

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# load api key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if api_key is None:
    st.error("GROQ_API_KEY not found. Please add it in .env file")
    st.stop()

# set api key so langchain can use it
os.environ["GROQ_API_KEY"] = api_key


# ---- page setup ----

st.set_page_config(page_title="DocuMind AI", page_icon="", layout="centered")

st.title("DocuMind AI")
st.subheader("Upload a document and ask questions about it")
st.write("Powered by Groq AI + LangChain + ChromaDB")


# ---- sidebar info ----

st.sidebar.header("How to use")
st.sidebar.write("1. Upload a .txt file")
st.sidebar.write("2. Wait for it to process")
st.sidebar.write("3. Type your question below")
st.sidebar.write("4. Get AI-powered answers")

st.sidebar.write("---")
st.sidebar.write("Built by Pragati Singh")


# ---- file upload section ----

uploaded_file = st.file_uploader("Upload your document (.txt)", type=["txt"])

if uploaded_file is not None:
    # read the uploaded file
    raw_text = uploaded_file.read()
    text = raw_text.decode("utf-8")

    # show file info to user
    file_name = uploaded_file.name
    file_size = len(text)
    st.info("File: " + file_name + " | Size: " + str(file_size) + " characters")

    # ---- split text into small chunks ----
    # we split because AI works better with small pieces of text
    # chunk_size = how big each piece is
    # chunk_overlap = how much overlap between pieces (so we dont lose context)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.create_documents([text])

    total_chunks = len(chunks)
    st.write("Document split into " + str(total_chunks) + " chunks for processing")

    # ---- create vector database ----
    # this converts text chunks into numbers (embeddings)
    # so we can search for similar text quickly
    # using huggingface model for this - its free and works offline

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # store chunks in chromadb vector database
    vectordb = Chroma.from_documents(chunks, embeddings)

    st.success("Document loaded and indexed successfully!")

    # ---- question answering section ----

    st.write("---")
    question = st.text_input("Ask a question about your document:")

    if question:
        # show loading spinner while AI is thinking
        with st.spinner("Searching document and generating answer..."):

            # step 1: find relevant chunks from vector db
            # k=3 means get top 3 most similar chunks
            results = vectordb.similarity_search(question, k=3)

            # step 2: combine the chunks into one context string
            context_parts = []
            for doc in results:
                context_parts.append(doc.page_content)

            context = "\n".join(context_parts)

            # step 3: send question + context to groq ai
            # using llama model because its fast and free on groq
            llm = ChatGroq(model="llama-3.3-70b-versatile")

            prompt = "Answer this question based on the given context only.\n\n"
            prompt = prompt + "Question: " + question + "\n\n"
            prompt = prompt + "Context: " + context + "\n\n"
            prompt = prompt + "If the answer is not in the context, say 'I could not find this information in the document.'"

            response = llm.invoke(prompt)
            answer = response.content

            # show the answer
            st.write("---")
            st.subheader("Answer:")
            st.write(answer)

            # show source chunks so user can verify
            # this is like showing your work in an exam haha
            with st.expander("View source chunks (for verification)"):
                for i in range(len(results)):
                    st.write("Chunk " + str(i + 1) + ":")
                    st.write(results[i].page_content)
                    st.write("---")

else:
    # show this when no file is uploaded yet
    st.write("---")
    st.write("Upload a .txt file to get started!")
    st.write("You can try with the sample.txt file included in the project.")
