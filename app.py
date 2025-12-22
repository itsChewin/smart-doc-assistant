import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. Header
st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.header("📄 Smart Document Assistant")

# 2. Sidebar 
with st.sidebar:
    st.title("Settings")
    language = st.radio("Select Answer Language:", ("English", "Thai"))
    
    # Pull API Key from .env first
    if os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.success("✅ API Key loaded from .env")
    else:
        # If not found, ask user to input
        api_key = st.text_input("Enter Google API Key:", type="password")

    uploaded_file = st.file_uploader("Upload PDF File", type="pdf")
    
    if st.button("Submit & Process"):
        if not api_key:
            st.error("Please enter your API Key")
        elif not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            with st.spinner("Processing..."):
                # Temporary save uploaded PDF
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # --- Data Pipeline ---
                # load PDF 
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                
                # split PDF into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)
                
                # transform to embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                
                # Create Database for searching (Vector Store)
                vector_store = FAISS.from_documents(final_documents, embeddings)
                vector_store.save_local("faiss_index")
                
                st.success("Done! You can now ask questions.")

# 3. Q&A Section
user_question = st.text_input("Ask a question about your PDF:")

if user_question and api_key:
    # --- check if database exists ---
    if os.path.exists("faiss_index"):
        # if exists, load the DB
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # search for related docs
        docs = new_db.similarity_search(user_question)

        # prompt template
        prompt_template = f"""
        You are an intelligent assistant for document analysis.
        
        Context:
        {{context}}
        
        Question:
        {{question}}

        Instructions:
        1. Answer purely based on the given Context.
        2. If the answer is not in the context, state "Information not found in the document."
        3. Keep the answer professional and concise.
        4. Answer in {language} language only.
        
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", google_api_key=api_key, temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        with st.spinner("Thinking..."):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("### Answer:")
            st.write(response["output_text"])
            
    else:
        # if not, prompt user to upload & process PDF first
        st.warning("⚠️ กรุณาอัปโหลดไฟล์ PDF และกดปุ่ม 'Submit & Process' ก่อนเริ่มถามคำถามครับ")