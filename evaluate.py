import json
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# reusing your existing functions from app.py
def getPDFText(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def getTextChunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def getVectorStore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def getChain(vectorstore):
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
    return chain, retriever

def main():
    print("Loading PDF...")
    text = getPDFText("syllabus.pdf")
    chunks = getTextChunks(text)
    vectorstore = getVectorStore(chunks)
    chain, retriever = getChain(vectorstore)

    print("Loading test set...")
    with open("test_set.json", "r") as f:
        test_set = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Running questions through chain...")
    for item in test_set:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # get retrieved chunks
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]

        # get answer from chain
        answer = chain.invoke({
            "input": question,
            "chat_history": []
        })

        print(f"Q: {question}")
        print(f"A: {answer}\n")

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    # build dataset for ragas
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("Running RAGAS evaluation...")
    llm_wrapper = LangchainLLMWrapper(ChatOpenAI())
    embeddings_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm_wrapper,
        embeddings=embeddings_wrapper
    )
    print("\n=== RESULTS ===")
    print(results)

if __name__ == "__main__":
    main()