# Import necessary modules
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
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI


# load env's vars from .env
load_dotenv()


# OpenAI Endpoint details - to be set in .env ------------------------------
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# --------------------------------------------------------------------------


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



# Data extraction!
def prepare_docs(pdf):
    """
    Llama Parse API to extract the content and metadata from the PDF files.
    To fix the document used in the chat the extracted content and metadata are then written to separate files in the script directory.
    """
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        # num_workers=4,
        verbose=True,
        language="en",
    )
    for doc in pdf:
        documents = parser.load_data(doc.name)

    content = documents[0].text
    print(content)
    script_dir = os.path.dirname(__file__)

    file_path = os.path.join(script_dir, "content.txt")
    with open(file_path, "w") as f:
        f.write(content)

    return content

# Chunki the documents
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


# Conversation!
def get_conversation_chain(vectordb):
    # using openai services
    llama_llm = ChatOpenAI()
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
        rephrase_question=False,  # huge problem
    )
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain


def handle_userinput(user_question):
    """
    Handle the userinput!
    """
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    sources = [doc.metadata for doc in response["source_documents"]]
    # -------------------- SOURCE ----------------------
    st.write("Sources:")
    for i, source in enumerate(sources):
        st.markdown(f"• Source {i+1}: {source}")
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

def main():
    load_dotenv()

    st.set_page_config(
        page_title="ALXGPT", page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("TalkToPDF :books:")
    user_question = st.text_input(
        "Ask a question about your documents"
    )

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        # st.write(split_docs)
        # st.write(content)
        # st.write(split_docs)
        # st.write(st.session_state)
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                content = prepare_docs(pdf_docs)
                st.write(content)
                # get the text chunks
                # split_docs = get_text_chunks(content)
                # create vector store
                # vectorstore = ingest_into_vectordb(split_docs)
                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()