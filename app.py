# File: app.py
import streamlit as st
import os
from llama_parse import LlamaParse
from dotenv import load_dotenv
from html_templates import css, bot_template, user_template
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Create media directory if it doesn't exist
MEDIA_DIR = os.path.join(os.getcwd(), 'media')
os.makedirs(MEDIA_DIR, exist_ok=True)

# Configure the system prompt
llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document.
Your responses should adhere to the following guidelines:
-----
Guidelines
-----
- Answer the question only based on the provided documents.
- Be direct and factual. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Do not fabricate information or include questions in your responses.
- Try to give all the information related to the question step by step
-------
Question: {question}
-------
[/INST]
"""

# Data extraction
def prepare_docs(pdf_files):
    """
    Llama Parse API to extract the content and metadata from the PDF files.
    To fix the document used in the chat the extracted content and metadata are then written to separate files in the script directory.
    """
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
        language="en",
    )
    contents = []
    for pdf in pdf_files:
        file_path = os.path.join(MEDIA_DIR, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        documents = parser.load_data(file_path)
        contents.append(documents[0].text)
        print(documents[0].text)

    return "\n\n".join(contents)

# Chunking the documents
def get_text_chunks(content):
    """
    Chunking component, using MarkdownHeaderSplitter and the RecursiveCharacterTextSplitter from Langchain
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(content)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.split_documents(md_header_splits)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs

# Ingest data
def ingest_into_vectordb(split_docs):
    """
    Data ingestion into vectordb
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db.save_local(DB_FAISS_PATH)
    print(f"Vector db is ready!")
    return db

# Conversation
def get_conversation_chain(vectordb):
    llama_llm = Ollama(model="llama3", base_url="http://34.71.171.183:11434/")
    retriever = vectordb.as_retriever()
    prompt = PromptTemplate.from_template(llmtemplate)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llama_llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        rephrase_question=False,
    )
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain

def handle_userinput(user_question):
    """
    Handle the user input
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            print(message.content)
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )

def intro_section():
    st.title("DocuBot :books:")
    st.image("https://wgmimedia.com/wp-content/uploads/2023/05/How-to-Talk-to-a-PDF-With-AI.jpg", use_column_width=True)
    st.write("Imagine effortlessly navigating through your PDF documents as if you were having a conversation. With DocuBot, you can now interact with your documents using natural language queries. Simply ask questions about the content, and receive detailed, contextually relevant answers directly from the document. This innovative approach transforms static PDFs into dynamic, responsive tools that enhance your understanding and streamline your workflow.")

def feature_section():
    st.header("Key Features")

    features = [
        {
            "name": "PDF Upload and Processing",
            "image": "images/pdf_upload.png",
            "description": "Easily upload your PDF documents for processing. The app extracts and processes the content for further analysis."
        },
        {
            "name": "Interactive Question-Answering",
            "image": "images/question_answer.png",
            "description": "Ask questions about your documents and receive accurate answers based on the content. Interact with your PDFs using natural language."
        },
        {
            "name": "Accurate Document Analysis",
            "image": "images/document_analysis.png",
            "description": "Benefit from precise and reliable analysis of your PDF documents, ensuring that you get the information you need."
        },
    ]

    for feature in features:
        st.subheader(feature["name"])
        st.image(feature["image"], use_column_width=True)
        st.write(feature["description"])

def about_section():
    st.header("About the Project")
    st.write("""
    The inspiration for the DocuBot project emerged from observing the transformative impact of AI across various domains. With AI becoming a ubiquitous presence, revolutionizing everything from customer service to creative arts, I was driven to explore how it could enhance one of the most ubiquitous and often overlooked tools in our daily lives: PDF documents.

    As I delved into this vision, I realized that while AI was making waves in areas like language translation and image recognition, PDFs remained static and largely unresponsive. The idea of making PDFs interactive and more accessible began to take shape. I envisioned a tool that would not only allow users to navigate through their documents more intuitively but also engage with the content in a meaningful way.

    This led me to conduct extensive research on integrating AI with PDF technologies. I explored various natural language processing techniques and machine learning models, aiming to develop a solution where users could interact with their documents as if they were having a conversation. The goal was clear: to create a tool that could understand questions about the content and provide insightful, detailed answers, thereby transforming the way people interact with their documents.

    The journey was both challenging and exhilarating. It required not just technical expertise but also a deep understanding of user needs and behaviors. The result is DocuBot—a project born from the convergence of AI advancements and the quest for a more interactive, user-friendly document experience. By harnessing the power of AI, DocuBot aims to enhance productivity and understanding, making every PDF a dynamic resource rather than a static file.
    
    This project is part of my portfolio for Holberton School. You can find the links to my profiles and the project repository below:
    """)
    st.write("[LinkedIn](https://www.linkedin.com/in/lokmane-rouijel/)")
    st.write("[GitHub](https://github.com/loki9919)")
    st.write("[Project Repository](https://github.com/loki9919/DocuBot)")

def main():
    load_dotenv()
    st.set_page_config(page_title="DocuBot", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Intro"

    pages = ["Intro", "Features", "About", "DocuBot"]
    selected_page = st.sidebar.selectbox("Navigate", pages)

    if selected_page:
        st.session_state.current_page = selected_page

    if st.session_state.current_page == "Intro":
        intro_section()
    elif st.session_state.current_page == "Features":
        feature_section()
    elif st.session_state.current_page == "About":
        about_section()
    elif st.session_state.current_page == "DocuBot":
        st.header("DocuBot 📚")
        user_question = st.text_input("Ask a question about your documents")

        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
            )
            if st.button("Process"):
                with st.spinner("Processing"):
                    content = prepare_docs(pdf_docs)
                    st.write(content)
                    # Get the text chunks
                    split_docs = get_text_chunks(content)
                    # Create vector store
                    vectorstore = ingest_into_vectordb(split_docs)
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
