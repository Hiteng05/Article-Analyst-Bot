from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
import pickle
import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.title("Analyst Bot : News Research Tool 📈")
st.sidebar.title("News Article URLs")

file_path = "faiss_db.pkl"

repoid = 'mistralai/Mistral-7B-Instruct-v0.3'
model = HuggingFaceEndpoint(repo_id=repoid,token=token,temperature=0.5)
embeddings = HuggingFaceEndpointEmbeddings()

urls = []
for i in range(1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=10000,
        chunk_overlap=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    vectorstore.save_local("vector_database")

def get_chain():
       prompt_template = """Answer the question from the provided context,
       make sure to provide all the details,if the answer is not present in the context,
       just say "Info not present",do not make up any information.\n\n
       Context:\n {context}?\n
       Question:\n {question}\n

       Answer:
       """

       
       prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
       chain = load_qa_chain(llm=model,chain_type="stuff",prompt=prompt)
       return chain


query =str(main_placeholder.text_input("Question: "))

if query:
        new_db = FAISS.load_local("vector_database",embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query)

        chain = get_chain()
        
        response = chain({"input_documents":docs,"question":query},return_only_outputs=True)
        st.header("Answer")
        st.write(response["output_text"])
        
