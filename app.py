# .\venv\Scripts\activate
# streamlit run app.py
    
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from htmlTemplates import css, bot_template, user_template

def getPDFText(pdfDocs):
    text = ""
    for pdf in pdfDocs:
        pdfReader = PdfReader(pdf)
        for page in pdfReader.pages:
            text += page.extract_text()
    return text


def getTextChunks(text):
    textSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = textSplitter.split_text(text)
    return chunks

def getVectorStore(textChunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=textChunks,embedding=embeddings)
    return vectorstore


def getConversationChain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the context below.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(doc.page_content for doc in retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def handleUserInput(userQuestion):
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    answer = st.session_state.conversation.invoke({
        "input": userQuestion,
        "chat_history": st.session_state.chat_history
    })

    st.session_state.chat_history.append(HumanMessage(content=userQuestion))
    st.session_state.chat_history.append(AIMessage(content=answer))

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chat Assistant", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    userQuestion = st.text_input("Ask a question about your documents:")
    
    if userQuestion:
        # Check if the conversation chain is actually ready
        if st.session_state.conversation is None:
            st.warning("⚠️ Please upload and process your PDFs before asking a question.")
        else:
            handleUserInput(userQuestion)

    #st.write(user_template.replace("{{MSG}}", "Hello PDF Chat Assistant!"), unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hi there! What would you like to know?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Manage Chat")
        if st.button("Reset Conversation"):
            st.session_state.conversation = None
            st.session_state.chatHistory = None
            st.rerun()
        
        st.subheader("Your documents")
        pdfDocs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdfDocs:
                st.warning("⚠️ Please upload a PDF file first.")
            else:
                with st.spinner("Processing"):
                    # get raw pdf text
                    rawText = getPDFText(pdfDocs)

                    # get the text chunks
                    textChunks = getTextChunks(rawText)

                    # create vector store
                    vectorstore = getVectorStore(textChunks)

                    # create conversation chain
                    st.session_state.conversation = getConversationChain(vectorstore)


if __name__ == '__main__':
    main()