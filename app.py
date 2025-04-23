import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# ✅ FAISS comes from langchain_community (moved)
from langchain_community.vectorstores import FAISS

# ✅ Google Generative AI Embeddings still from langchain_google_genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ✅ Google Chat Model
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # ✅ Recommended modern QA chain
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found in .env file")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to generate FAISS index from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Get the RetrievalQA chain
def get_qa_chain(vector_store):
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, "answer is not available in the context", 
    don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# Function to handle user question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        qa_chain = get_qa_chain(new_db)
        response = qa_chain.invoke({"query": user_question})  # ✅ Replaces deprecated __call__
        st.write("Reply: ", response["result"])
    except Exception as e:
        st.error(f"Error during the search or answer generation: {str(e)}")

# Streamlit app main function
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    # User input section
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if not pdf_docs:
            st.warning("Please upload at least one PDF file.")

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete. Now you can ask questions.")
                    else:
                        st.error("No text extracted from the uploaded PDFs.")
            else:
                st.error("Please upload PDFs before processing.")

if __name__ == "__main__":
    main()
