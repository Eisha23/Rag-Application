import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# Streamlit UI setup
st.title("RAG Application")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embedded" not in st.session_state:
    st.session_state.embedded = False

uploaded_files = st.file_uploader("Upload your files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files and not st.session_state.embedded:
    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split(".")[-1]) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Load with appropriate loader
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(tmp_file_path)
        else:
            continue

        docs = loader.load()
        all_docs.extend(docs)

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)

    # Save to ChromaDB
    vectorstore = Chroma.from_documents(
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
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use only the context below to answer the question.
        Context:
        {context}
        Question:
        {question}
        """
    )

    # Generate response
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": context, "question": query})

    # Save to chat history
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", response))
  
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