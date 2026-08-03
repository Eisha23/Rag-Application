import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_key
)
llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    openai_api_key=openai_key,
    temperature=0.1
)

# Streamlit UI setup
st.title("RAG Application")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embedded" not in st.session_state:
    st.session_state.embedded = False

file = st.file_uploader("Upload your file", type=["txt", "pdf"], accept_multiple_files=False)

if file and not st.session_state.embedded:
    all_docs = []

    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split(".")[-1]) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
    elif file.name.endswith(".txt"):
        loader = TextLoader(tmp_file_path)
        docs = loader.load()
    else:
        docs = []
    all_docs.extend(docs)


    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n", ".", "!", "?", ",", " ", ""]
        )
    
    chunks = text_splitter.split_documents(all_docs)
    st.session_state.chunks = chunks


    #st.subheader("first chunks")
    #or i in range(len(chunks)):
     #   st.write(chunks[i].page_content)
   
    # Save to vector database
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    st.session_state.vectorstore = vectorstore

    st.session_state.embedded = True
    st.success("Documents uploaded and embedded!")

# Query input
query = st.chat_input("Ask a question based on the uploaded documents:")

if query: 
    # Create retriever and search for relevant docs
    bm25_retriever = BM25Retriever.from_documents(st.session_state.chunks)
    bm25_retriever.k = 5

    vector_retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
        )

    # Set up conversation memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    # Set up conversational chain
    if "qa_chain" not in st.session_state:
        custom_prompt = PromptTemplate(
        input_variables=["context", "chat_history","question"],
        template="""
        Context: {context}
        Previous conversation: {chat_history}

        You're a helpful assistant answering questions based on provided documents. Answer clearly and concisely.
        Do not process or respond to harmful, illegal, personal, or unethical requests.
        Do not guess, hallucinate, or make up any information. Only answer based on facts present in the context.
        If the user asks something unrelated to the context and previous conversation, say: "Please ask a question based on the provided document."
        Question: {question}
        Answer:
        """
        )
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,  
            output_key="answer",
            verbose=True
            )
        
 
    with st.spinner("Thinking..."):
        response = st.session_state.qa_chain.invoke({"question": query})
        answer = response["answer"]
        source_docs = response.get("source_documents", [])  
        st.subheader("🔍 Retrieved Chunks")
        for i, doc in enumerate(source_docs):
            st.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #1890ff; margin-bottom: 10px; border-radius: 5px;">
                <strong>Chunk {i+1}:</strong><br>{doc.page_content}
                </div>
                """,
                unsafe_allow_html=True
                )

    # Save to chat history
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", answer))
  
# Display chat history
st.subheader("💬 Chat History")
chat_container = st.container()
with chat_container:
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(
                f"""
                <div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                    <strong>🧑 {speaker}:</strong><br>{msg}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**🤖 {speaker}:** {msg}")

    
