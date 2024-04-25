import streamlit as st
import nltk
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage

class RAGSystem:
    def __init__(self, pdf_path, api_key_path):
        self.pdf_path = pdf_path
        self.api_key = self._load_api_key(api_key_path)
        self.loader = PyPDFLoader(self.pdf_path)
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.data = self.loader.load_and_split(self.text_splitter)
        nltk.download("punkt")
        self.text_splitter_nltk = NLTKTextSplitter(chunk_size=5000, chunk_overlap=500)
        self.chunks = self.text_splitter_nltk.split_documents(self.data)
        self.embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key, model="models/embedding-001")
        genai.configure(api_key=self.api_key, transport='rest')
        self.db = self._build_db()
        self.db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=self.embedding_model)
        self.retriever = self.db_connection.as_retriever(search_kwargs={"k": 5})
        self.chat_model = ChatGoogleGenerativeAI(google_api_key=self.api_key, model="gemini-1.5-pro-latest")
        self.chat_template = self._build_chat_template()
        self.output_parser = StrOutputParser()
        self.rag_chain = self._build_rag_chain()

    def _load_api_key(self, api_key_path):
        with open(api_key_path, 'r') as f:
            return f.read().strip()

    def _build_db(self):
        db = Chroma.from_documents(self.chunks, self.embedding_model, persist_directory="./chroma_db_")
        db.persist()
        return db

    def _build_chat_template(self):
        return ChatPromptTemplate.from_messages([
            # System Message Prompt Template
            SystemMessage(content="""You are a Helpful AI Bot.
            Welcome to the RAG System for "Leave No Context Behind" Paper.
            As a RAG AI, your task is to provide insightful answers based on the context provided."""),
            # Human Message Prompt Template
            HumanMessagePromptTemplate.from_template("""Please provide an answer to the question below, considering the context of the "Leave No Context Behind" paper:
            Context:
            {context}

            Question:
            {question}

            Answer: """)
        ])

    def _build_rag_chain(self):
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.chat_template
            | self.chat_model
            | self.output_parser
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_answer(self, question):
        return self.rag_chain.invoke(question)

# Streamlit app
st.title('Ask Me Anything')


st.markdown("""
    ### Ask Me Anything: Using Leave No Context Behind paper by Google
    
    You're now connected to an AI assistant powered by the Retrieval-Augmented Generation (RAG) system.
    Welcome to the "Leave No Context Behind" Q&A platform.

    As an AI, I'm here to assist you in finding insightful answers. Simply provide some context or ask your question, and I'll do my best to provide relevant and helpful responses.

    """)

# Input field for the question
question_input = st.text_area("Enter your question:", value='')

# Initialize the RAG system
pdf_path = "Leave_no_context_behind.pdf"
api_key_path = "Key/.gemini_api_key.txt"
rag_system = RAGSystem(pdf_path, api_key_path)

# Button to generate answer
if st.button("Get Answers"):
    answer = rag_system.get_answer(question_input)
    st.markdown(f"**Answer:** {answer}")
