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

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Smart Document Assistant", layout="wide")
st.header("📄 Smart Document Assistant (Powered by Gemini)")

# 2. Sidebar สำหรับใส่ API Key และ Upload File
with st.sidebar:
    st.title("Settings")
    language = st.radio("Select Answer Language:", ("English", "Thai"))
    
    # ลองดึง Key จาก .env ก่อน
    if os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("GOOGLE_API_KEY")
        st.success("✅ API Key loaded from .env")
    else:
        # ถ้าไม่มีใน .env ค่อยให้กรอกเอง
        api_key = st.text_input("Enter Google API Key:", type="password")

    uploaded_file = st.file_uploader("Upload PDF File", type="pdf")
    
    if st.button("Submit & Process"):
        if not api_key:
            st.error("Please enter your API Key")
        elif not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            with st.spinner("Processing..."):
                # Save file ชั่วคราวเพื่ออ่าน
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # --- ส่วนสำคัญ: Data Pipeline ---
                # อ่านไฟล์ PDF
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                
                # หั่นไฟล์เป็นชิ้นเล็กๆ (Chunks)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)
                
                # แปลงข้อความให้เป็น Vector (Embeddings) ด้วย Gemini
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                
                # สร้าง Database สำหรับค้นหา (Vector Store)
                vector_store = FAISS.from_documents(final_documents, embeddings)
                vector_store.save_local("faiss_index")
                
                st.success("Done! You can now ask questions.")

# 3. ส่วนถาม-ตอบ (User Interface)
user_question = st.text_input("Ask a question about your PDF:")

if user_question and api_key:
    # --- ส่วนที่แก้: เช็คก่อนว่ามี Database หรือยัง ---
    if os.path.exists("faiss_index"):
        # ถ้ามีไฟล์แล้ว ค่อยโหลด
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # ค้นหาเนื้อหาที่เกี่ยวข้อง
        docs = new_db.similarity_search(user_question)

        # ... (ส่วน Prompt Template และ Gemini เหมือนเดิม) ...
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
        # ถ้ายังไม่มีไฟล์ ให้เตือนดีๆ
        st.warning("⚠️ กรุณาอัปโหลดไฟล์ PDF และกดปุ่ม 'Submit & Process' ก่อนเริ่มถามคำถามครับ")